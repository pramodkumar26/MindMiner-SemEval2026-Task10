# SemEval-2026 Task 10: Multi-Model Approaches to Conspiracy Detection and Psycholinguistic Marker Extraction

Code and results from our work on SemEval-2026 Task 10 (PsyCoMark), where we tried to detect conspiratorial thinking and extract psycholinguistic markers from Reddit comments.

## What we did

The task has two parts. Subtask 1 asks you to find spans of five marker types in a comment: Actor, Action, Effect, Victim, and Evidence. Subtask 2 is simpler on paper, just a binary call on whether the comment shows conspiratorial thinking.

We tried five models and submitted to both subtasks. Our best run was DeBERTa-v3-large, which finished 2nd on Subtask 2 with 0.80 weighted F1, and 13th on Subtask 1 with 0.16 macro F1.

## Models we tested

The five models cover a range of sizes and approaches:

1. DistilBERT-base (66M parameters), the lightweight baseline using the organizer scripts as is.
2. DeBERTa-v3-base (184M parameters), a mid-sized model with disentangled attention.
3. DeBERTa-v3-large (435M parameters), which ended up being our best model overall.
4. RoBERTa-large (355M parameters), included to see if BERT-style pretraining tweaks helped.
5. LLaMA-3.1-8B with LoRA, where we only train 0.5% of the parameters and use 4-bit quantization to fit it on a single GPU.

## Results

### Subtask 1: Marker Extraction (Macro Overlap F1)

| Model | Test F1 | Notes |
|-------|---------|-------|
| DeBERTa-v3-large | 0.16 |
| DeBERTa-v3-base | 0.15 | |
| DistilBERT | 0.15 | Organizer baseline |
| RoBERTa-large | 0.14 | |
| LLaMA-3.1-8B (LoRA) | 0.14 | |

The per-marker breakdown for DeBERTa-v3-large is worth looking at because the gap is huge: Actor 0.364, Effect 0.209, Action 0.123, Evidence 0.094, Victim 0.000. Victim collapsed to zero. We talk about why in the paper.

### Subtask 2: Conspiracy Detection (Weighted F1)

| Model | Test F1 | Notes |
|-------|---------|-------|
| DeBERTa-v3-large | 0.80  |
| LLaMA-3.1-8B (LoRA) | 0.80 | |
| RoBERTa-large | 0.79 | |
| DeBERTa-v3-base | 0.75 | |
| DistilBERT | 0.75 | Organizer baseline |

## Setup

### Getting the data

Reddit's data policy means the dataset comes redacted. You have to rehydrate it yourself.

1. Clone the official starter pack:
   ```bash
   git clone https://github.com/hide-ous/semeval26_task10_starter_pack.git
   cd semeval26_task10_starter_pack
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Grab `train_redacted.jsonl` from [Zenodo](https://doi.org/10.5281/zenodo.15114171) and put it in the folder.

4. Run the rehydration script:
   ```bash
   python rehydrate_data.py
   ```
   You should end up with `train_rehydrated.jsonl` containing the actual comment text.

### Preprocessing

We dropped samples labeled "Can't tell" so the binary task could focus on clear conspiracy versus non-conspiracy cases. After filtering we had 4,316 training samples. We trained on all of them and used the official dev set for model selection rather than carving out yet another internal validation split. Test scores came from Codabench submissions.

## Training approaches

### Subtask 2 (Detection)

The encoder models (DistilBERT, DeBERTa, RoBERTa) used a linear classification head with cross-entropy loss, nothing fancy. LLaMA was different. We framed it as a generative task with a prompt asking for a "Yes" or "No" answer.

Hyperparameters varied by model size. Batch sizes from 2 to 64, learning rates from 1e-5 to 1e-4, training between 5 and 10 epochs depending on convergence behavior. The bigger models needed gradient checkpointing to fit in memory.

### Subtask 1 (Extraction)

We tried three different formulations because no single one was obviously right:

1. Question answering for DistilBERT, where each marker type becomes its own QA task using `DistilBertForQuestionAnswering`.
2. Token classification for DeBERTa and RoBERTa, with simplified BIO tagging and one model trained per marker type, so five models total per architecture.
3. A generative approach for LLaMA, where one model produces all five marker spans as structured text.

The per-marker approach was simpler to reason about than a multi-label tagger but it does mean training five models. The LLaMA setup handled all five at once but needed careful prompt engineering to keep the output format consistent.

## What we learned

A few things stood out from the experiments.

Conspiracy detection turned out to be reasonably tractable. All five models landed between 0.75 and 0.80 F1, which is a tight spread for models ranging from 66M to 8B parameters. Architecture choice and training quality mattered more than raw scale.

Marker extraction was a different story. Our best score was 0.16 macro F1, which is a fivefold ratio away from detection. The challenge isn't model capacity, it's the task itself. Spans overlap, references are often implicit, and some marker types lack clear syntactic boundaries.

The per-marker variation was the most interesting finding. Actor at 0.364 F1 versus Victim at 0.000 F1 isn't just a difficulty gradient, it points at fundamentally different problem structures. Actors are usually explicit noun phrases like "the government" or "corporations." Victims are often pronouns like "us" or "the people" scattered across sentences. A span tagger has nothing to grab onto for victims.

LoRA worked well. LLaMA-3.1-8B matched DeBERTa-v3-large on detection while only training 41.9M parameters out of 8 billion. With 4-bit quantization the whole thing fits on a single GPU.

## Hardware

We trained on Google Colab Pro for most runs. The encoder models (DistilBERT, DeBERTa, RoBERTa) ran on a T4 with 16GB. LLaMA-3.1-8B with LoRA needed a larger GPU and 4-bit quantization to fit.

Reported training times from the paper:

* DistilBERT: about 10 minutes for Subtask 2, about 28 minutes for Subtask 1
* DeBERTa-v3-large: 22.2 minutes for Subtask 2

The other times depend on hardware availability and we haven't reported them in the paper. See the paper for the full hyperparameter tables.

## Submitting to Codabench

1. Train your model and generate predictions on the test set.
2. Format predictions per the task specification. Pay attention to the `conspiracy` field. It needs to be a string `"Yes"` or `"No"`. Null values cause submission failures.
3. Save as `submission.jsonl`.
4. Zip it: `submission.zip`.
5. Upload to the right competition page:
   * Subtask 1 (Marker Extraction): https://www.codabench.org/competitions/10751/
   * Subtask 2 (Conspiracy Detection): https://www.codabench.org/competitions/10749/
6. Wait for evaluation, then add your result to the leaderboard.

## Future work

A few directions we think are worth chasing:

1. Figuring out why Victim detection collapses to 0.000 F1 while Actor works reasonably well. The answer probably isn't more data or a bigger model.
2. Better span boundary handling for overlapping and nested markers.
3. Marker-specific architectures rather than a single uniform approach. Evidence behaves differently from Actor, and treating them the same is likely costing us.
4. Joint training across detection and extraction. Right now we treat them independently.
5. Discourse-level modeling for the markers that span multiple sentences. Standard transformers with fixed context windows aren't great at this.

## Limitations

A few honest caveats about what we did:

* The marker distribution is heavily skewed across categories. Models handle markers with clear syntactic boundaries reasonably well but struggle on the more implicit ones.
* Reddit comments have a lot of noise. Sarcasm, slang, broken syntax, and inline links all make conspiratorial reasoning harder to detect, even for large models.
* Several marker types (especially Evidence and Effect) span multiple sentences and need long-range context that our models don't really capture.
* Our results come from single-seed runs. We didn't average across seeds, so some of the small differences between models could shift if we did.

## A note on LLaMA

LLaMA-3.1-8B is under Meta's community license, not a fully open license. You'll need to accept Meta's terms before you can download the weights. The encoder models (DistilBERT, DeBERTa, RoBERTa) are available through HuggingFace under permissive licenses.

## Acknowledgments

Thanks to the task organizers, Mattia Samory, Felix Soldner, and Veronika Batzdorfer, for releasing the PsyCoMark benchmark and putting together the starter pack.

## License

MIT.

## Contact

* Pramod Kumar Ajmeera: pramod.ajmeera@colorado.edu
* Akshara Sri Lakshmipathy: akshara.lakshmipathy@colorado.edu

University of Colorado Boulder
