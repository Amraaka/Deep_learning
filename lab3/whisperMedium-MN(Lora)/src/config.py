"""Training hyperparameters and model ids (shared by run_train.py and gradio_demo.py)."""

MODEL_NAME = "openai/whisper-medium"
LANGUAGE = "Mongolian"
LANGUAGE_ABBR = "mn"
TASK = "transcribe"
SEED = 42
MAX_AUDIO_SEC = 30.0
MIN_AUDIO_SEC = 1.0
MAX_TARGET_TOKENS = 448

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

WARMUP_STEPS = 50
GENERATION_MAX_LENGTH = 225
