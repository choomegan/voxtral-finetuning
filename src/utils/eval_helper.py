import os
import torch

from peft import PeftModel
from transformers import VoxtralForConditionalGeneration

from utils.custom_model import (
    VoxtralForConditionalGenerationWithLID,
    VoxtralWithTaskTokenRouting,
)
from utils.constants import SRCLANG2ID


def load_model_for_evaluation(config, device, logger):
    """
    Load model from checkpoint, handling:
    - Task routing wrapper
    - LoRA adapters
    - LID head weights

    Returns:
        Tuple of (model, is_task_routing)
    """
    checkpoint_path = config.checkpoint_path

    # Detect checkpoint type
    task_routing_marker = os.path.join(checkpoint_path, "task_routing.txt")
    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    lid_head_file = os.path.join(checkpoint_path, "lid_head.pt")

    is_task_routing = os.path.exists(task_routing_marker)
    has_lora = os.path.exists(adapter_config)
    has_lid_head = os.path.exists(lid_head_file)

    logger.info("=" * 60)
    logger.info("Checkpoint Analysis:")
    logger.info(f"  📂 Path: {checkpoint_path}")
    logger.info(f"  🔀 Task routing: {is_task_routing}")
    logger.info(f"  🧩 LoRA adapter: {has_lora}")
    logger.info(f"  🏷️  LID head: {has_lid_head}")
    logger.info("=" * 60)

    # =========================
    # 1. Load Base Model
    # =========================
    if has_lid_head:
        logger.info("Loading VoxtralForConditionalGenerationWithLID...")
        base_model = VoxtralForConditionalGenerationWithLID.from_pretrained(
            config.model,  # Always load from base model path
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            num_languages=len(SRCLANG2ID),
        )
    else:
        logger.info("Loading standard VoxtralForConditionalGeneration...")
        base_model = VoxtralForConditionalGeneration.from_pretrained(
            config.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    base_model.to(device)

    # =========================
    # 2. Load LoRA Adapter
    # =========================
    if has_lora:
        logger.info(f"Loading LoRA adapter from {checkpoint_path}...")
        base_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logger.info("✅ LoRA adapter loaded")

        # Check if LID head was saved in adapter (via modules_to_save)
        adapter_state = base_model.state_dict()
        lid_keys = [k for k in adapter_state.keys() if "lid_head" in k]
        if lid_keys:
            logger.info(f"✅ Found {len(lid_keys)} LID head params in adapter")

    # =========================
    # 3. Load LID Head (separate file)
    # =========================
    if has_lid_head:
        logger.info(f"Loading LID head from {lid_head_file}...")

        # Navigate to the actual lid_head module
        if hasattr(base_model, "base_model"):
            # Unwrap PEFT model: PeftModel.base_model.model.lid_head
            target_lid_head = base_model.base_model.model.lid_head
        else:
            # Direct model
            target_lid_head = base_model.lid_head

        # Load state dict
        lid_state = torch.load(lid_head_file, map_location=device)
        target_lid_head.load_state_dict(lid_state)
        logger.info(
            f"✅ Loaded LID head ({os.path.getsize(lid_head_file) / 1e6:.1f}MB)"
        )

    # =========================
    # 4. Wrap with Task Routing (if needed)
    # =========================
    if is_task_routing:
        logger.info("Wrapping with VoxtralWithTaskTokenRouting...")

        # Get hidden size from correct location
        if hasattr(base_model, "base_model"):
            # PEFT wrapped
            hidden_size = base_model.base_model.model.audio_tower.config.hidden_size
        else:
            hidden_size = base_model.audio_tower.config.hidden_size

        model = VoxtralWithTaskTokenRouting(
            base_model=base_model,
            num_languages=len(SRCLANG2ID),
            hidden_size=hidden_size,
        )
        model.to(device)
        logger.info("✅ Task routing wrapper applied")
    else:
        model = base_model

    # =========================
    # 5. Set to Eval Mode
    # =========================
    model.eval()

    # Print model structure for debugging
    logger.info("\n📊 Model Structure:")
    if is_task_routing:
        logger.info("  VoxtralWithTaskTokenRouting")
        logger.info("  └── base_model (PeftModel)" if has_lora else "  └── base_model")
        if has_lora:
            logger.info("      └── VoxtralForConditionalGenerationWithLID")
    else:
        logger.info("  PeftModel" if has_lora else "  VoxtralForConditionalGeneration")
        if has_lora and has_lid_head:
            logger.info("  └── VoxtralForConditionalGenerationWithLID")

    logger.info("✅ Model loaded successfully\n")

    return model, is_task_routing
