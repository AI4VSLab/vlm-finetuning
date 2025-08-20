import torch
import torch.nn as nn
from data import DataCollator
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, SiglipVisionModel, TextStreamer
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)

class AdaptiveNormalizationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        return self.activation(x)

class VisionLanguageModel:
    def __init__(self, config, device):
        logger.info("Creating model...")
        self.config = config
        self.device = device

        # Hugging Face model id
        model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`

        # Check if GPU benefits from bfloat16
        if torch.cuda.get_device_capability()[0] < 8:
            raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

        # Define model init arguments
        model_kwargs = dict(
            attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
            torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
            device_map="auto", # Let torch decide how to load the model
        )

        # BitsAndBytesConfig int-4 config
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )

        # Load model and tokenizer
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

        # Add adaptive normalization layer
        # self.siglip_vision = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        # siglip_output_dim = self.siglip_vision.config.hidden_size
        siglip_output_dim = 1152  # From your SigLIP checkpoint shapes
        gemma_vision_dim = model.vision_tower.config.hidden_size  # Gemma 3's expected dimension
        self.adaptive_norm = AdaptiveNormalizationLayer(siglip_output_dim, gemma_vision_dim)

        # Load the trained SigLIP image encoder
        siglip_checkpoint_path = config.siglip_checkpoint_path  # From config.yaml instead of hardcoded path
        siglip_checkpoint = torch.load(siglip_checkpoint_path, map_location=device)

        # DEBUG: Print available keys
        print("Available keys in checkpoint:")
        print(list(siglip_checkpoint.keys()))

        # # Extract only vision/image encoder parts from model_state_dict
        # full_state_dict = siglip_checkpoint['model_state_dict']
        # vision_state_dict = {k.replace('image_encoder.', ''): v for k, v in full_state_dict.items() if 'image_encoder' in k}
        # if vision_state_dict:
        #     model.vision_tower.load_state_dict(vision_state_dict, strict=False)
        # else:
        #     print("No image encoder found in checkpoint, skipping SigLIP loading")
        
        # Create separate SigLIP vision model for feature extraction
        self.siglip_vision = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        siglip_full_state_dict = siglip_checkpoint['model_state_dict']
        siglip_vision_state_dict = {k.replace('image_encoder.', ''): v for k, v in siglip_full_state_dict.items() if 'image_encoder' in k}
        if siglip_vision_state_dict:
            self.siglip_vision.load_state_dict(siglip_vision_state_dict, strict=False)
        else:
            print("No image encoder found in checkpoint, skipping SigLIP loading")

        # Replace Gemma's vision tower (image encoder) with the trained SigLIP encoder
        # model.vision_tower.load_state_dict(siglip_checkpoint['image_encoder_state_dict'])

        # Freeze the image encoder if you don't want to fine-tune it further
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        
        print(f"[INFO] Loaded SigLIP encoder from {siglip_checkpoint_path} and replaced Gemma's vision tower (image encoder).")

        # Store original vision tower forward
        original_vision_forward = model.vision_tower.forward

        def custom_vision_forward(pixel_values, **kwargs):
            # Use SigLIP for feature extraction
            siglip_features = self.siglip_vision(pixel_values).last_hidden_state
            # Apply adaptive normalization
            adapted_features = self.adaptive_norm(siglip_features)
            return type('VisionOutput', (), {'last_hidden_state': adapted_features})()

        model.vision_tower.forward = custom_vision_forward

        processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=[
                "lm_head",
                "embed_tokens",
            ],
        )
        peft_model = get_peft_model(model, peft_config)

        self.prompt = config.prompt
        self.model = peft_model
        self.processor = processor
        self.data_collator = DataCollator(self.processor)

        logger.info("Model created.")

    def train_mode(self):
        self.model.train()

    def inference_mode(self):
        self.model.eval()

    def generate(self, image, use_streamer=True):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.prompt}, {"type": "image"}],
            }
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        if use_streamer:
            text_streamer = TextStreamer(self.processor, skip_prompt=True)
            self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                                use_cache=True, temperature=1.0, top_p=0.95, top_k=64)
            return "" # Streamer prints, so return empty string or None
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=128,
                                            use_cache=True, temperature=1.0, top_p=0.95, top_k=64)
            # Decode only the newly generated tokens (excluding the prompt)
            decoded_output = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return decoded_output
