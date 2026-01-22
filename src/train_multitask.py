"""
Multitask finetuning for ASR,, ST, T2T, and LID
"""

import logging
import os
from collections import Counter
from datetime import datetime

import torch
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    AutoTokenizer,
)

from utils.collators import (
    StreamingASRCollator,
    StreamingMultiTaskCollator,
    StreamingSTCollator,
    StreamingT2TCollator,
    StreamingLIDCollator,
)
from utils.dataset_utils import (
    load_preprocessed_multitask_dataset,
    compute_lid_class_weights,
)
from utils.train_utils import SafeTrainer
from utils.custom_model import (
    VoxtralWithTaskTokenRouting,
    VoxtralForConditionalGenerationWithLID,
)
from utils.constants import SRCLANG2ID

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbose logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # module-level logger


def create_model(config, processor, device):
    """
    Create model with optional task token routing and LoRA.

    Returns either:
    - VoxtralWithTaskTokenRouting (with LoRA applied to base model)
    - VoxtralForConditionalGeneration (with LoRA applied)
    """
    logger.info("Loading base model...")

    # =========================
    # 1. Load base Voxtral
    # =========================
    # Load with LID head if task routing is enabled
    num_languages = len(SRCLANG2ID) if config.tasks.lid.enabled else None

    if config.trainer.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = VoxtralForConditionalGeneration.from_pretrained(
            config.model,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
    else:
        # if LID is included in the tasks
        if config.tasks.lid.enabled:
            base_model = VoxtralForConditionalGenerationWithLID.from_pretrained(
                config.model,
                torch_dtype=torch.bfloat16 if config.trainer.bf16 else torch.float16,
                num_languages=num_languages,
            )
        else:
            base_model = VoxtralForConditionalGeneration.from_pretrained(
                config.model,
                torch_dtype=torch.bfloat16 if config.trainer.bf16 else torch.float16,
            )

    # =========================
    # 2. Apply LoRA (BEFORE wrapping)
    # =========================
    if getattr(config, "lora", None):
        logger.info("🧩 Applying LoRA adapters")

        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias="none",
            target_modules=list(config.lora.target_modules),
            task_type="SEQ_2_SEQ_LM",
        )

        base_model = get_peft_model(base_model, lora_config)
        base_model.enable_input_require_grads()

        try:
            base_model.print_trainable_parameters()
        except Exception:
            pass

    # =========================
    # 3. Task-token routing wrapper
    # =========================
    if config.trainer.use_task_routing:
        logger.info("🔀 Using task routing approach")

        model = VoxtralWithTaskTokenRouting(
            base_model=base_model,
            lid_class_weights=lid_class_weights,
            use_focal_loss=config.tasks.lid.focal_loss.enabled,
            focal_gamma=config.tasks.lid.focal_loss.gamma,
        )

        logger.info("✅ Initialized task routing model")
    else:
        logger.info("📝 Using original prompt-based approach (no task tokens)")
        model = base_model

    # NEED TO UNFREEZE TASK SPECIFI HEADS! Peft will freeze the whole base model automatically
    if hasattr(model, "base_model") and hasattr(model.base_model, "lid_head"):
        for p in model.base_model.lid_head.parameters():
            p.requires_grad = True
        logger.info("✅ LID head is trainable")

    # =========================
    # 4. Freeze audio encoder
    # =========================
    audio_encoder = (
        model.base_model.audio_tower
        if hasattr(model, "base_model")
        else model.audio_tower
    )
    for param in audio_encoder.parameters():
        param.requires_grad = False
    logger.info("✅ Froze audio encoder")

    return model


def _initialize_collators(config, processor, model_id: str):
    """
    Initialize only the collators needed based on enabled tasks.
    """
    tasks_config = config.get("tasks", {})

    collators = {}

    # ASR collator
    if tasks_config.get("asr", {}).get("enabled", False):
        asr_config = tasks_config["asr"]
        collators["asr"] = StreamingASRCollator(
            processor=processor,
            model_id=model_id,
            sample_rate=16000,
            lang=None,  # Will use per-sample language from manifest
        )
        logger.info("✅ Initialized ASR collator")

    # ST collator
    if tasks_config.get("s2tt", {}).get("enabled", False):
        st_config = tasks_config["s2tt"]
        collators["st"] = StreamingSTCollator(
            processor=processor,
            model_id=model_id,
            incl_src_lang=st_config.get("incl_src_lang", True),
        )
        logger.info("✅ Initialized ST collator")

    # T2T collator
    if tasks_config.get("t2t", {}).get("enabled", False):
        collators["t2t"] = StreamingT2TCollator(
            processor=processor,
            model_id=model_id,
        )
        logger.info("✅ Initialized T2T collator")

    if tasks_config.get("lid", {}).get("enabled", False):
        collators["lid"] = StreamingLIDCollator(
            processor=processor,
            model_id=model_id,
        )
        logger.info("✅ Initialized LID collator")

    multitask_collator = StreamingMultiTaskCollator(
        asr_collator=collators.get("asr"),
        st_collator=collators.get("st"),
        t2t_collator=collators.get("t2t"),
        lid_collator=collators.get("lid"),
    )
    logger.info("✅ Initialized multi-task collator")
    return multitask_collator


def main():
    accelerator = Accelerator()
    config = OmegaConf.load("config/train_multitask.yaml")

    # --- Setup WandB ---
    if config.exp_manager.logger == "wandb" and accelerator.is_main_process:
        if config.trainer.resume_from_checkpoint and config.exp_manager.wandb.run_id:
            wandb.init(
                project=config.exp_manager.wandb.project,
                id=config.exp_manager.wandb.run_id,
                resume="must",
            )
        else:
            wandb.init(
                project=config.exp_manager.wandb.project,
                name=config.exp_manager.name,
            )
    # --- Experiment name ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.exp_manager.name}_{timestamp}"
    output_dir = os.path.join(config.exp_manager.exp_dir, exp_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Processor ---
    logger.info("Loading processor...")
    processor = VoxtralProcessor.from_pretrained(config.model)
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    processor.tokenizer = tokenizer

    # Load preprocessed datasets (this caches to disk)
    logger.info("Loading or creating preprocessed datasets...")
    (
        train_dataset,
        eval_dataset_asr,
        eval_dataset_st,
        eval_dataset_t2t,
        eval_dataset_lid,
    ) = load_preprocessed_multitask_dataset(
        train_manifest=config.data.train_manifest,
        eval_manifest=config.data.eval_manifest,
        incl_asr=config.tasks.asr.enabled,
        incl_s2tt=config.tasks.s2tt.enabled,
        incl_t2t=config.tasks.t2t.enabled,
        incl_lid=config.tasks.lid.enabled,
    )

    logger.info(
        "Eval dataset sizes — ASR: %s, ST: %s, T2T: %s, LID: %s",
        len(eval_dataset_asr) if eval_dataset_asr else "N/A",
        len(eval_dataset_st) if eval_dataset_st else "N/A",
        len(eval_dataset_t2t) if eval_dataset_t2t else "N/A",
        len(eval_dataset_lid) if eval_dataset_lid else "N/A",
    )

    # --- Compute language weights ---
    if config.trainer.lang_class_weighting:
        logger.info(
            "---------------- Adding language weighting to loss function -------------"
        )
        records = train_dataset.to_list()
        lang_counts = Counter([r["source_lang"] for r in records])
        logger.info("Language counts: %s", lang_counts)

        total = sum(lang_counts.values())
        lang_freq = {lang: count / total for lang, count in lang_counts.items()}

        # Inverse frequency
        lang_weights = {lang: 1.0 / freq for lang, freq in lang_freq.items()}

        # Optional: normalize weights to sum to 1
        norm_factor = sum(lang_weights.values())
        lang_weights = {lang: w / norm_factor for lang, w in lang_weights.items()}
        logger.info("Normalized language weights: %s", lang_weights)

    # --- LID classification class weighting ---
    if config.tasks.lid.enabled and (
        config.tasks.lid.focal_loss.enabled or config.trainer.focal_loss.enabled
    ):
        logger.info("✅ Adding class weighting for LID task")
        lid_class_weights = compute_lid_class_weights(
            train_dataset, method="inverse_freq"
        )
    else:
        lid_class_weights = None

    # --- Model ---
    model = create_model(config, processor, device)

    # --- Collators ---
    multi_collator = _initialize_collators(
        config,
        processor=processor,
        model_id=config.model,
    )

    # --- Training ---
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.trainer.train_batch_size,
        per_device_eval_batch_size=config.trainer.eval_batch_size,
        gradient_accumulation_steps=config.trainer.grad_accum,
        learning_rate=config.trainer.lr,
        num_train_epochs=config.trainer.epochs,
        warmup_steps=config.trainer.warmup_steps,
        bf16=config.trainer.bf16,
        logging_steps=config.trainer.logging_steps,
        eval_on_start=config.trainer.eval_on_start,
        eval_steps=config.trainer.eval_steps,
        save_steps=config.trainer.save_steps,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=config.trainer.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_combined_loss",
        greater_is_better=False,
        report_to=config.exp_manager.logger,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        # dataloader_prefetch_factor=2,  # Prefetch 2 batches per worker
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        lr_scheduler_type="cosine",
        seed=3407,
        ddp_find_unused_parameters=True,
    )

    trainer = SafeTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset={
            "asr": eval_dataset_asr,
            "st": eval_dataset_st,
            "t2t": eval_dataset_t2t,
            "lid": eval_dataset_lid,
        },
        data_collator=multi_collator,
        lang_weight_map=lang_weights if config.trainer.lang_class_weighting else None,
        use_uncertainty=config.trainer.task_uncertainty_weighting,
        use_pcgrad=config.trainer.use_pcgrad,
        pcgrad_reduction=config.trainer.pcgrad_reduction,
    )

    logger.info("Starting multi-task training...")
    trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    logger.info("Training complete!")

    trainer.save_model()
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
