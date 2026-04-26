MindMiner at SemEval-2026 Task 10: Multi-Model Approaches to Conspiracy Detection and Psycholinguistic Marker Extraction
This repository contains the code and experimental results for our participation in SemEval-2026 Task 10 (PsyCoMark), focused on psycholinguistic conspiracy marker extraction and detection from Reddit comments.
Overview
We benchmarked five transformer architectures across the two official subtasks:
Subtask 1 (Marker Extraction): span extraction for five psycholinguistic markers — Actor, Action, Effect, Victim, and Evidence.
Subtask 2 (Conspiracy Detection): binary classification to identify conspiratorial thinking in Reddit comments.
Our best model (DeBERTa-v3-large) reached 0.80 weighted F1 on Subtask 2 (2nd place) and 0.16 macro F1 on Subtask 1 (13th place).
Models
DistilBERT-base (66M parameters): lightweight baseline using the organizer scripts.
DeBERTa-v3-base (184M parameters): mid-sized model with disentangled attention.
DeBERTa-v3-large (435M parameters): our best performing model overall.
RoBERTa-large (355M parameters): BERT variant with optimized pretraining.
LLaMA-3.1-8B (8B parameters, 0.5% trainable with LoRA): parameter-efficient large language model fine-tuning.
Results
Subtask 1: Marker Extraction (Macro Overlap F1)
Model	Test F1	Notes
DeBERTa-v3-large	0.16	13th place
DeBERTa-v3-base	0.15	
DistilBERT	0.15	Organizer baseline
RoBERTa-large	0.14	
LLaMA-3.1-8B (LoRA)	0.14	
Per-marker breakdown for DeBERTa-v3-large: Actor 0.364, Effect 0.209, Action 0.123, Evidence 0.094, Victim 0.000.
Subtask 2: Conspiracy Detection (Weighted F1)
Model	Test F1	Notes
DeBERTa-v3-large	0.80	2nd place
LLaMA-3.1-8B (LoRA)	0.80	
RoBERTa-large	0.79	
DeBERTa-v3-base	0.75	
DistilBERT	0.75	Organizer baseline
Setup
Getting the data
The dataset requires rehydration because of Reddit's data policies.
Clone the official starter pack:
```bash
   git clone https://github.com/hide-ous/semeval26_task10_starter_pack.git
   cd semeval26_task10_starter_pack
   ```
Install dependencies:
```bash
   pip install -r requirements.txt
   ```
Download `train_redacted.jsonl` from Zenodo and place it in the folder.
Rehydrate the data:
```bash
   python rehydrate_data.py
   ```
This produces `train_rehydrated.jsonl` containing the full comment text.
Data preprocessing
We filtered out samples labeled "Can't tell" so that binary classification could focus on clear conspiracy versus non-conspiracy cases. After filtering, the training set contained 4,316 samples. We trained on this full set and used the official development set for model selection and hyperparameter tuning rather than carving out an additional internal validation split. Test scores were obtained by submitting to Codabench.
Training approaches
Subtask 2: Binary classification
The encoder models (DistilBERT, DeBERTa, RoBERTa) used a linear classification head with cross-entropy loss. LLaMA was framed as a generative classification task with instruction prompts asking for a "Yes" or "No" response.
Hyperparameters varied by model size. Batch sizes ranged from 2 to 64, learning rates from 1e-5 to 1e-4, and training ran for 5 to 10 epochs. Gradient checkpointing was used for the larger models to fit within GPU memory.
Subtask 1: Marker extraction
We tested three different formulations:
Question answering (DistilBERT): each marker type framed as a separate QA task using `DistilBertForQuestionAnswering`.
Token classification (DeBERTa, RoBERTa): five separate models with simplified BIO tagging, one model per marker type.
Generative (LLaMA): a single model generating all marker spans as structured text.
The separate-model-per-marker approach simplified handling overlapping spans but required training five models instead of one. The generative LLaMA approach handled all five markers in a single model at the cost of more careful prompt engineering.
Key findings
Conspiracy detection is feasible. Models reached 0.75 to 0.80 F1, showing transformers can pick up on conspiratorial framing reasonably well.
Marker extraction remains hard. Best result was 0.16 macro F1, a fivefold ratio compared to detection. The difficulty stems from overlapping span boundaries, implicit references, and subtle distinctions between marker types.
Model scale shows diminishing returns. Going from 66M to 8B parameters improved Subtask 2 by only 0.05 F1. Architecture and training quality matter more than raw size.
Per-marker performance varies dramatically. Actor detection reaches 0.364 F1 because actors are typically explicit noun phrases; Victim detection completely fails (0.000 F1) because victims are often implicit or pronominal references scattered across sentences.
LoRA enables efficient large-LLM fine-tuning. LLaMA-3.1-8B matched DeBERTa-v3-large performance on Subtask 2 while training only 0.5% of its parameters (41.9M out of 8B), with 4-bit quantization keeping the model in single-GPU memory.
Hardware
Training was conducted primarily on Google Colab Pro with the following GPU configurations:
T4 (16GB) for the encoder models (DistilBERT, DeBERTa, RoBERTa).
A larger GPU with 4-bit quantization for LLaMA-3.1-8B.
Approximate training times reported in the paper:
DistilBERT: about 10 minutes (Subtask 2), about 28 minutes (Subtask 1).
DeBERTa-v3-large: 22.2 minutes (Subtask 2).
Other times depend on hardware availability and are not reported here. See the paper for full hyperparameter tables.
Submission
Train your model and generate predictions.
Format predictions according to the task specification.
Create `submission.jsonl` with the required fields. Note that the `conspiracy` field must be a valid `"Yes"` or `"No"` string; null values cause submission failures on Codabench.
Zip the file: `submission.zip`.
Upload to Codabench:
Subtask 1 (Marker Extraction): https://www.codabench.org/competitions/10751/
Subtask 2 (Conspiracy Detection): https://www.codabench.org/competitions/10749/
Wait a few minutes for evaluation and add the result to the leaderboard.
Future work
Our analysis suggests several directions worth pursuing:
Investigate Victim detection failure. Why does this marker collapse to 0.000 F1 when Actor reaches 0.364?
Better span boundary handling for overlapping and nested markers.
Marker-specific architectures. Different markers may need different modeling strategies rather than a uniform approach.
Multi-task learning. Joint training across detection and extraction may help.
Discourse-level modeling. Evidence and Effect markers often span multiple sentences and need richer context modeling than fixed-window transformers provide.
Limitations
The marker distribution is skewed across categories, and per-marker results show large variation in difficulty (F1 ranging from 0.000 for Victim to 0.364 for Actor). Models handle markers with clear syntactic boundaries reasonably well while struggling on more implicit categories.
The Reddit domain contains heavy linguistic noise such as sarcasm, slang, broken syntax, and inline hyperlinks. Even large models have trouble untangling this noise when the underlying task is conspiratorial reasoning.
Many markers, especially Evidence and Effect, span multiple sentences. Standard transformers with fixed context windows do not capture these cross-sentence dependencies well.
Results are based on single-seed runs; we did not perform multi-seed averaging.
Reproducibility note
LLaMA-3.1-8B is distributed under Meta's community license. You will need to accept Meta's terms before downloading the model weights. The encoder models (DistilBERT, DeBERTa, RoBERTa) are available under permissive licenses through HuggingFace.
Acknowledgments
We thank the task organizers (Mattia Samory, Felix Soldner, and Veronika Batzdorfer) for releasing the PsyCoMark benchmark and the starter pack scripts.
License
This repository is released under the MIT License.
Contact
For questions or collaboration:
Pramod Kumar Ajmeera: pramod.ajmeera@colorado.edu
Akshara Sri Lakshmipathy: akshara.lakshmipathy@colorado.edu
University of Colorado Boulder
