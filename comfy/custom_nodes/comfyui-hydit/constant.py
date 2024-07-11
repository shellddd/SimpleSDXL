import os
from .hydit.constants import SAMPLER_FACTORY

base_path = os.path.dirname(os.path.realpath(__file__))
HUNYUAN_PATH = os.path.join(base_path, "..", "..", "models",  "checkpoints")
SCHEDULERS_hunyuan = list(SAMPLER_FACTORY.keys())


T5_PATH = os.path.join(base_path, "..", "..", "models",  "llms")
LORA_PATH = os.path.join(base_path, "..", "..", "models",  "loras")
