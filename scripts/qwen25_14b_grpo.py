import os
from dotenv import load_dotenv, find_dotenv
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
import re

from trl import GRPOConfig, GRPOTrainer

_ = load_dotenv(find_dotenv())

os.environ["HF_ACCESS_TOKEN"] = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
max_seq_length = 2048 #Increase this for longer reasoning traces
lora_rank =32 #larger ranks are smarter but slower

def main(repo_name: str, output_name: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-14B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.8, #reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank, #8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth", #enable long context finetuning
        random_state=2025
    )

    # Load and prep dataset
    SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

    XML_COT_FORMAT = """\
    <reasoning>
    {reasoning}
    </reasoning>
    <answer>
    {answer}
    </answer>
    """

    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    # uncomment middle messages for 1-shot prompting
    def get_gsm8k_questions(split = "train") -> Dataset:
        data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
        data = data.map(lambda x: { # type: ignore
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        }) # type: ignore
        return data # type: ignore

    dataset = get_gsm8k_questions()

    # Reward functions
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def int_reward_func(completions, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1])*0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count

    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]
    
    training_args = GRPOConfig(
        # use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 8, # Decrease if out of memory
        max_prompt_length = 256,
        max_completion_length = 200,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 250,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        run_name=output_name,
        output_dir = output_name,
    )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()
    
    print("\n\n TRAINING COMPLETE \n\n")
    model.push_to_hub_merged(
        repo_name,
        tokenizer, 
        save_method = "merged_16bit",
        token = os.environ["HF_ACCESS_TOKEN"]
    )
    
    print("\n\n MERGED MODEL PUSHED TO HUGGINGFACE \n\n")

if __name__ == "__main__":
    repo_name = "tituslhy/qwen25_14b_grpo_t1"
    output_name="qwen25_14b_grpo_take1"
    main(repo_name=repo_name, output_name=output_name)
    