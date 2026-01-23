# SemEval 2026 Task 10: PsyCoMark System Paper

This repository contains code and results for our participation in SemEval 2026 Task 10, focused on psycholinguistic conspiracy marker extraction and detection from Reddit comments.

## Overview

We benchmarked five transformer architectures on two subtasks:
- **Task A (Conspiracy Detection)**: Binary classification to identify conspiratorial thinking
- **Task B (Marker Extraction)**: Span extraction for five psycholinguistic markers (Actor, Action, Effect, Victim, Evidence)

Our best model (DeBERTa-v3-large) achieved 0.80 F1 on Task A (2nd place) and 0.16 F1 on Task B (13th place).

## Models Tested

1. **DistilBERT-base** (66M parameters) - Lightweight baseline using organizer scripts
2. **DeBERTa-v3-base** (184M parameters) - Mid-sized model with disentangled attention
3. **DeBERTa-v3-large** (435M parameters) - Our best performing model overall
4. **RoBERTa-large** (355M parameters) - BERT variant with optimized pretraining
5. **LLaMA-3.1-8B** (8B parameters, 0.5% trainable with LoRA) - Parameter-efficient LLM fine-tuning

## Results Summary

### Task A: Conspiracy Detection (Weighted F1)
| Model | Test F1 | Rank |
|-------|---------|------|
| DeBERTa-v3-large | 0.80 | 2nd |
| LLaMA-3.1-8B (LoRA) | 0.80 | - |
| RoBERTa-large | 0.79 | - |
| DeBERTa-v3-base | 0.75 | - |
| DistilBERT | 0.75 | Baseline |

### Task B: Marker Extraction (Macro F1)
| Model | Test F1 | Rank |
|-------|---------|------|
| DeBERTa-v3-large | 0.16 | 13th |
| DeBERTa-v3-base | 0.15 | - |
| DistilBERT | 0.15 | Baseline |
| RoBERTa-large | 0.14 | - |
| LLaMA-3.1-8B (LoRA) | 0.14 | - |

## Setup Instructions

### Getting the Data

The dataset requires rehydration because of Reddit's data policies:

1. Clone the starter pack repository:
   ```bash
   git clone https://github.com/hide-ous/semeval26_starter_pack.git
   cd semeval26_starter_pack
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download `train_redacted.jsonl` from [Zenodo](https://doi.org/10.5281/zenodo.15114171) and place it in the folder

4. Rehydrate the data:
   ```bash
   python rehydrate_data.py
   ```
   This creates `train_rehydrated.jsonl` with the full comment text.

### Data Preprocessing

We filtered out samples labeled "Can't tell" to focus on clear conspiracy vs. non-conspiracy cases. After filtering, we split the data:
- Training: 3,177 samples (90%)
- Validation: 354 samples (10%)
- Test: Evaluated through CodaBench

## Training Approaches

### Task A: Binary Classification

All encoder models (DistilBERT, DeBERTa, RoBERTa) used a linear classification head with cross-entropy loss. LLaMA used a generative approach with instruction prompts asking for "Yes" or "No" responses.

Key hyperparameters varied by model size:
- Batch sizes: 2-64 (smaller for larger models)
- Learning rates: 1e-5 to 1e-4
- Epochs: 5-10
- Gradient checkpointing for memory efficiency

### Task B: Marker Extraction

We tested three different formulations:
1. **Question Answering** (DistilBERT): Treated each marker type as a separate QA task
2. **Token Classification** (DeBERTa, RoBERTa): Five separate models with simplified BIO tagging, one per marker
3. **Generative** (LLaMA): Single model generating all marker spans as structured text

The separate-model-per-marker approach simplified handling overlapping spans but required training five models instead of one.

## Key Findings

1. **Conspiracy detection is feasible**: Models achieved 0.75-0.80 F1, showing that transformers can effectively identify conspiratorial patterns.

2. **Marker extraction remains challenging**: Best result was only 0.16 F1, representing a 5x performance gap compared to detection. The task difficulty stems from overlapping boundaries, implicit references, and subtle distinctions between marker types.

3. **Model scale shows diminishing returns**: Going from 66M to 8B parameters improved Task A by only 0.05 F1. Architecture matters more than raw size.

4. **Per-marker performance varies dramatically**: Actor detection (F1: 0.36) works because actors are explicit noun phrases, while Victim detection (F1: 0.00) fails completely due to implicit references.

5. **LoRA enables efficient LLM fine-tuning**: LLaMA-3.1-8B matched DeBERTa-v3-large performance while training only 0.5% of parameters (41.9M out of 8B).

## Hardware Requirements

Training was conducted on:
- T4 GPU (16GB) for encoder models (DistilBERT, DeBERTa, RoBERTa)
- A100 GPU for LLaMA-3.1-8B with 4-bit quantization

Approximate training times on T4:
- DistilBERT: 10 min (Task A), 28 min (Task B)
- DeBERTa-v3-large: 22 min (Task A), 136 min (Task B)
- RoBERTa-large: 50 min (Task A), 90 min (Task B)

## Submission Process

1. Train your model and generate predictions
2. Format predictions according to task specifications
3. Create `submission.jsonl` with predictions
4. Zip the file: `submission.zip`
5. Upload to CodaBench:
   - Task A: https://www.codabench.org/competitions/10749/
   - Task B: https://www.codabench.org/competitions/10751/
6. Wait for evaluation (usually a few minutes)
7. Add results to leaderboard

## Future Work

Our analysis suggests several directions for improvement:

1. **Investigate Victim detection failure**: Why does this marker achieve 0.00 F1 when Actor achieves 0.36?
2. **Better span boundary handling**: Current methods struggle with overlapping and nested markers
3. **Marker-specific architectures**: Different markers may need different modeling approaches
4. **Multi-task learning**: Joint training on detection and extraction might help
5. **Discourse-level modeling**: Evidence and Effect markers often span multiple sentences



## Contact

For questions or collaboration:
- Akshara Sri Lakshmipathy: akla8196@colorado.edu
- Pramod Kumar Ajmeera: pramod.ajmeera@colorado.edu

University of Colorado Boulder
