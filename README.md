# Finetune Voxtral for ASR/ Speech Translation with Transformers

## Set up

Using UV

```
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
```

# Multi Task
### Dataset Format

```
{
    "source":
        {
            "text": "CEO LTAT",             # for t2t
            "lang": "zsm",
            "audio_local_path": "audio/TeF4KD586kk-254.wav",
            "sampling_rate": 16000
        },
    "target":
        {
            "text": "the CEO of LTAT.",     # for st
            "lang": "eng"}
        },
    "audio_filepath": "audio/audio_1.wav",  # for asr
    "text": "this is a transcript"
}
```
### Training

Edit training config under config/train_multitask.yaml

To run the training script (single gpu):
```
uv run src/train_multitask.py
```


To run the training script (multi-gpu):
```
accelerate launch --multi_gpu --num_processes 2 src/train_multitask.py 
```
### Evaluation
Tasks are evaluated separately using `eval_asr.py` and `eval_st.py`

Edit eval config under `config/eval_asr.yaml` and `config/eval_st.yaml`, before running the evaluation scripts.



# ASR

### Dataset Format

Standard NeMo manifest format with `audio_filepath` and `text` fields.

```
{
    "audio_filepath": "audio/audio_1.wav",
    "duration": 5.038,
    "start": 1166.599,
    "end": 1171.637,
    "text": "this is a transcript"
}
```

### Training

Edit training config under config/train_asr.yaml

```
uv run src/train_asr.py
accelerate launch --multi_gpu --num_processes 2 src/train_asr.py 

```

### Evaluation

Edit eval config under config/eval_asr.yaml

```
uv run eval_asr.py
```

### Post-eval computation of WER

For mms dataset, normalization needs to be done to the texts before computing WER.

Run:

```
python3 metric_computation/compute_wer_norm.py --manifest /dir/eval_results.json
```

To remove <UNK> from references (if any), run:

```
python3 metric_computation/compute_wer_norm.py --manifest /dir/eval_results.json --remove_unk
```

# Speech Translation

### Dataset Format

Standard NeMo manifest format with nested fields of `source` and `target`.

```
{
    "source":
        {
            "text": "CEO LTAT",
            "lang": "zsm",
            "audio_local_path": "audio/TeF4KD586kk-254.wav",
            "sampling_rate": 16000
        },
    "target":
        {
            "text": "the CEO of LTAT.",
            "lang": "eng"}
        }
}
```

### Training

Edit training config under config/train_st.yaml

```
uv run src/train_st.py
```

### Evaluation

Edit eval config under config/eval_st.yaml

```
uv run eval_st.py
```

### Computation of other translation metrics

To compute xCOMET scores, visit https://huggingface.co/Unbabel/XCOMET-XL to ask for access.

```
python3 metric_computation/compute_mt_metrics.py --manifest /dir/eval_results.json
```


