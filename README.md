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

```
uv run train_lora.py
```
