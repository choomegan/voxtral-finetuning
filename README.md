# Finetune Voxtral for ASR with Transformers

## Set up

Using UV

```
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Dataset Format

Standard NeMo manifest format.

## Training
Edit training config under utils/eval.yaml

```
uv run train_lora.py
```

## Evaluation
Edit eval config under utils/eval.yaml
```
uv run eval_lora.py
```
This evaluation script will reference config.json for the evaluation 

### Post-eval computation of WER
For mms dataset, normalization needs to be done to the texts before computing WER.

Run:
```
python3 utils/norm_compute_wer.py --manifest /dir/eval_results.json
```

To remove <UNK> from references (if any), run:
```
python3 utils/norm_compute_wer.py --manifest /dir/eval_results.json --remove_unk
```