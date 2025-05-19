# Compact-LLMs-for-Clinical-Diagnostics-A-CoT-Guided-Approach

This repository contains two Jupyter notebooks that demonstrate end-to-end fine-tuning, evaluation, and inference of transformer-based language models on a medical dataset:

- **`Copy_of_finalmedicalGemma2_(9B)_Alpaca.ipynb`**  
  - Fine-tunes the 9-billion-parameter Gemma2 model on a custom medical Q&A corpus, using the Alpaca instruction-tuning recipe.  
  - Implements data preprocessing, prompt formatting, and training loop with gradient checkpointing.  
  - Tracks training losses and generates example medical Q&A at checkpoints.  
  - Exports the final model and tokenizer for inference.

- **`Copy_of_finalcotllamaformedical.ipynb`**  
  - Fine-tunes a Code-Tuned LLaMA (“CoT-Llama”) variant on the same medical corpus, leveraging chain-of-thought prompting.  
  - Includes custom CoT prompt templates, multi-step rationales, and contrastive loss for improved reasoning.  
  - Logs chain-of-thought outputs alongside final answers during evaluation.  
  - Exports a runnable inference demo with CoT capabilities.

---

## Contents

- `Copy_of_finalmedicalGemma2_(9B)_Alpaca.ipynb` – Gemma2–Alpaca medical fine-tuning  
- `Copy_of_finalcotllamaformedical.ipynb` – CoT-LLaMA medical fine-tuning  
- `README.md` – this file

---

## Prerequisites

- Python 3.8+  
- [PyTorch 2.x](https://pytorch.org/) (with CUDA support for GPU)  
- [Transformers](https://github.com/huggingface/transformers) >= 4.30  
- [Datasets](https://github.com/huggingface/datasets)  
- `peft`, `accelerate`, `bitsandbytes`  
- `wandb` (optional, for logging)  

Install with:

```bash
pip install torch torchvision transformers datasets peft accelerate bitsandbytes wandb
