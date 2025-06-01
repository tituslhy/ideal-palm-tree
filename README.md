# ideal-palm-tree
The LLM Finetuning Playground repository.

This GitHub repo is a companion resource to the articles:
1. [How to Train Your LLM: Teaching Toothless to Bite](https://medium.com/@tituslhy/how-to-train-your-llm-teaching-toothless-to-bite-8d9f56fe4b2a)
2. [How to RAFT your LLM: Retrieval Augmented Finetuning using Unsloth!](https://medium.com/@tituslhy/how-to-raft-your-llm-retrieval-augmented-finetuning-using-unsloth-4c3844a9a6e3)
3. [How to train your LLM to reason like DeepSeek: GRPO reinforcement learning using Unsloth!](https://medium.com/mitb-for-all/how-to-train-your-llm-to-reason-grpo-reinforcement-learning-using-unsloth-64af5e82ac3c)

<p align="center">
    <img src="./images/training_a_dragon.png">
</p>

## Content
In our notebooks folder, we discuss how to develop a training dataset and finetune an LLM in different ways - supervised finetuning and retrieval augmented finetuning.
```
.
| - notebooks
|   - 1a. training_dataset_gen.ipynb        <- Codes to generate our training dataset
|   - 1b. finetune_llama32_1bn.ipynb        <- Codes to finetune Llama 3.2 1bn!
|   - 2. llama32_1bn_RAFT.ipynb             <- Codes to finetune Llama 3.2 1bn using the RAFT recipe!
|   - 3a. llama32_1b_grpo.ipynb             <- Codes to finetune Llama 3.2 1bn using GRPO!
|   - 3b. qwen25_14b_grpo.ipynb             <- Codes to finetune Qwen 2.5 14bn using GRPO!
```

## Setup
```
uv sync
```
