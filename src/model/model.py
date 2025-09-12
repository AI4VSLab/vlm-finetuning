"""
Isolated RetinaVLM - Standalone version extracted from SpecialistVLMs research repo
"""

import torch
import numpy as np
from PIL import Image
from types import SimpleNamespace
import os
import sys

class DictNamespace:
    """Config object that supports both dictionary and attribute access"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

class RetinaVLMModel:
    def __init__(self, device="auto"):
        self.device = self._get_device(device)
        self.model = None

        # Add current directory to path for model imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)

        self._load_model()

    def _get_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _create_enhanced_config(self):
        """Create enhanced config with all required attributes"""
        config = SimpleNamespace()

        # Model config
        config.model = SimpleNamespace()
        config.model.checkpoint_path = None

        # Vision encoder config
        config.model.vision_encoder = DictNamespace(
            name='pretrained_resnet',
            feature_dim=2048,
            concatenate_tokens=False,
            unfrozen_layers=[],
            blocks=[2, 2, 2, 2],  # ResNet18 structure
            checkpoint=None
        )

        # Language model config - add initialize attribute
        config.model.language_model = SimpleNamespace()
        config.model.language_model.model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
        config.model.language_model.load_in_8bit = False
        config.model.language_model.load_in_4bit = True
        config.model.language_model.initialize = True  # Initialize the language model

        # Miniformer config
        config.model.miniformer = SimpleNamespace()
        config.model.miniformer.name = None
        config.model.miniformer.linear_upsample = False

        # Token configs
        config.model.p_tokens = 0
        config.model.s_tokens = 0
        config.model.image_patch_position_embedding = False

        # Dataset config
        config.dataset = SimpleNamespace()
        config.dataset.task = SimpleNamespace()
        config.dataset.task.max_txt_len = 512
        config.dataset.n_vision_tokens = 256

        # Device config
        config.devices = 1

        # Directory paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config.retfound_dir = current_dir
        config.flamingo_dir = current_dir
        config.llava_med_dir = current_dir
        config.octlatent_dir = current_dir
        config.pretrained_model_dir = current_dir

        return config

    def _load_model(self):
        try:
            print("Loading RetinaVLM model...")

            from models.retinavlm_wrapper import RetinaVLMConfig, RetinaVLM

            # Load HuggingFace config
            hf_config = RetinaVLMConfig.from_pretrained(
                "RobbieHolland/RetinaVLM",
                subfolder="RetinaVLM-Specialist"
            )

            # Create enhanced config and apply all attributes
            research_config = self._create_enhanced_config()

            # Apply all config attributes to the HF config
            for attr_name in dir(research_config):
                if not attr_name.startswith('_'):
                    setattr(hf_config, attr_name, getattr(research_config, attr_name))

            # Load the model
            self.model = RetinaVLM.from_pretrained(
                "RobbieHolland/RetinaVLM",
                subfolder="RetinaVLM-Specialist",
                config=hf_config
            ).eval()

            print(f"RetinaVLM loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading RetinaVLM: {e}")
            import traceback
            traceback.print_exc()
            raise

    def generate(self, prompt, image):
        instruction = prompt
        if instruction is None:
            instruction = ("Write an extensive report describing the OCT image and "
                         "listing any visible biomarkers or other observations. "
                         "Do not provide disease stage or patient referral recommendations yet.")

        try:
            # Use the model's forward method
            reports = self.model.forward([image], [instruction])
            return reports[0] if reports else "No report generated"

        except Exception as e:
            return f"Error generating report: {e}"
