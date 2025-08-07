import torch
from data import DataCollator
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)


class VisionLanguageModel:
    def __init__(self, config, device):
        logger.info("Creating model...")
        self.config = config
        self.device = device

        # Hugging Face model id
        model_id = config.model_id # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`

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
