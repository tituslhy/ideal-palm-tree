import os
from dotenv import load_dotenv, find_dotenv
from unsloth import FastLanguageModel
from datasets import load_dataset
import re

from trl import GRPOConfig, GRPOTrainer

_ = load_dotenv(find_dotenv())

def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

os.environ["HF_ACCESS_TOKEN"] = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
max_seq_length = 2048 #Increase this for longer reasoning traces
lora_rank =32 #larger ranks are smarter but slower

def main(repo_name: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "meta-llama/Llama-3.2-1B-Instruct",
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

    reasoning_start = "<start_working_out>"
    reasoning_end   = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    system_prompt = \
    f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""

    dataset = load_dataset("openai/gsm8k", "main", split = "train")
    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })
    
    match_format = re.compile(
        rf"^[\s]{{0,}}"\
        rf"{reasoning_start}.+?{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    
    match_numbers = re.compile(
        solution_start + r".*?([\d\.\,]{1,})",
        flags = re.MULTILINE | re.DOTALL
    )
    
    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Match if format is seen exactly!
            if match_format.search(response) is not None: score += 3.0
            scores.append(score)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Count how many keywords are seen - we penalize if too many!
            # If we see 1, then plus some points!
            score += 0.5 if response.count(reasoning_start) == 1 else -1.0
            score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
            score += 0.5 if response.count(solution_start)  == 1 else -1.0
            score += 0.5 if response.count(solution_end)    == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := match_format.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(0)
                continue
            # Correct answer gets 3 points!
            if guess == true_answer:
                score += 3.0
            # Match if spaces are seen, but less reward
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                # We also reward it if the answer is close via ratios!
                # Ie if the answer is within some range, reward it!
                try:
                    ratio = float(guess) / float(true_answer)
                    if   ratio >= 0.9 and ratio <= 1.1: score += 1.0
                    elif ratio >= 0.8 and ratio <= 1.2: score += 0.5
                    else: score -= 1.5 # Penalize wrong answers
                except:
                    score -= 1.5 # Penalize
            scores.append(score)
        return scores
    
    global PRINTED_TIMES
    PRINTED_TIMES = 0
    global PRINT_EVERY_STEPS
    PRINT_EVERY_STEPS = 5

    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := match_numbers.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        # Print only every few steps
        global PRINTED_TIMES
        global PRINT_EVERY_STEPS
        if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
            print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        PRINTED_TIMES += 1

        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                # Remove commas like in 123,456
                guess       = float(guess.strip().replace(",", ""))
                scores.append(1.5 if guess == true_answer else -0.5)
            except:
                scores.append(0)
                continue
        return scores
    
    max_prompt_length = 287 + 1 # + 1 just in case!
    training_args = GRPOConfig(
        learning_rate = 5e-6,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        num_train_epochs = 3, # Set to 1 for a full training run
        # max_steps = 500,
        max_grad_norm = 1.0,
        run_name="grpo_take8",
        report_to = "wandb", 
        output_dir = "grpo_outputs_take8",
    )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args = training_args,
        train_dataset = dataset
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
    
    # ① Point at the real binary in build/bin
    real_q = os.path.expanduser("~/llama.cpp/build/bin/llama-quantize")
    assert os.path.exists(real_q), f"{real_q} not found!"

    # ② Make a local 'llama.cpp' folder in your notebook working directory
    cwd = os.getcwd()
    local_pack = os.path.join(cwd, "llama.cpp")
    os.makedirs(local_pack, exist_ok=True)

    # ③ Symlink it as 'llama-quantize' and also as 'quantize'
    for name in ("llama-quantize", "quantize"):
        link = os.path.join(local_pack, name)
        if os.path.exists(link) or os.path.islink(link):
            os.remove(link)
        os.symlink(real_q, link)

    # ④ Verify
    print("Scripts folder sees:", os.listdir(local_pack))
    
    model.push_to_hub_gguf(
        repo_name, # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = os.environ["HUGGINGFACE_ACCESS_TOKEN"], # Get a token at https://huggingface.co/settings/tokens
    )
    
    print("\n\n GGUF MODEL PUSHED TO HUGGINGFACE \n\n")

if __name__ == "__main__":
    repo_name = "tituslhy/grpo_llama32_1b"
    main(repo_name=repo_name)
    