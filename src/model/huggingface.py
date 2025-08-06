import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import TextStreamer
from PIL import Image
from typing import Any, Dict, List

class HuggingFaceVLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def process_vision_info(self, messages: list[dict]) -> list[Image.Image]:
        image_inputs = []
        # Iterate through each conversation
        for msg in messages:
            # Get content (ensure it's a list)
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]

            # Check each content element for images
            for element in content:
                if isinstance(element, dict) and (
                    "image" in element or element.get("type") == "image"
                ):
                    # Get the image and convert to RGB
                    if "image" in element:
                        image = element["image"]
                    else:
                        image = element
                    image_inputs.append(image.convert("RGB"))
        return image_inputs

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = []
        images = []
        for example in examples:
            image_inputs = self.process_vision_info(example["messages"])
            text = self.processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(image_inputs)

        # Tokenize the texts and process the images
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Mask image tokens
        # Prioritize getting the image token ID from the tokenizer's special tokens map
        # Fallback to the hardcoded value (262144) if "<image>" token is not explicitly defined.
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")

        # Mask tokens for not being used in the loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch


class HuggingFaceVLM:
    def __init__(self, config, device):
        self.device = device
        self.model_name = config.model_name
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
        model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
        processor = AutoProcessor.from_pretrained(self.model_name)

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

        self.model = peft_model
        self.processor = processor
        self.data_collator = HuggingFaceVLMDataCollator(processor)
        self.prompt = model.prompt


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
            text=input_text,
            images=[image], # Pass image as a list, as expected by the processor
            return_tensors="pt",
            add_special_tokens=False, # Ensure special tokens are not added twice if already in chat template
        ).to(self.device)

        if use_streamer:
            text_streamer = TextStreamer(self.processor.tokenizer, skip_prompt=True) # Use tokenizer from processor
            self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                                use_cache=True, temperature=1.0, top_p=0.95, top_k=64)
            return "" # Streamer prints, so return empty string or None
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=128,
                                            use_cache=True, temperature=1.0, top_p=0.95, top_k=64)
            # Decode only the newly generated tokens (excluding the prompt)
            decoded_output = self.processor.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return decoded_output
