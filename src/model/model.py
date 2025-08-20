from unsloth import FastVisionModel
from transformers import TextStreamer
import logging

logger = logging.getLogger(__name__)


class VisionLanguageModel:
    def __init__(self, config):
        logger.info("Creating model...")
        self.config = config
        self.load()
        self.system_prompt = self.config.system_prompt
        self.user_prompt = self.config.user_prompt
        logger.info("Model created.")

    def save(self):
        self.model.save_pretrained(self.config.save_path)
        self.processor.save_pretrained(self.config.save_path)

    def load(self):
        if self.config.from_checkpoint:
            model, processor = FastVisionModel.from_pretrained(
                model_name=self.config.checkpoint_path,  # YOUR MODEL YOU USED FOR TRAINING
                load_in_4bit=True,  # Set to False for 16bit LoRA
            )
        else:
            model, processor = FastVisionModel.from_pretrained(
                self.config.name,
                load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
                use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
            )
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=True,  # False if not finetuning vision layers
                finetune_language_layers=True,  # False if not finetuning language layers
                finetune_attention_modules=True,  # False if not finetuning attention layers
                finetune_mlp_modules=True,  # False if not finetuning MLP layers
                r=self.config.lora.r,  # The larger, the higher the accuracy, but might overfit
                lora_alpha=self.config.lora.alpha,  # Recommended alpha == r at least
                lora_dropout=self.config.lora.dropout,
                bias="none",
                random_state=3407,
                use_rslora=False,  # We support rank stabilized LoRA
                loftq_config=None,  # And LoftQ
                target_modules="all-linear",  # Optional now! Can specify a list if needed
                modules_to_save=[
                    "lm_head",
                    "embed_tokens",
                ],
            )
        self.model = model
        self.processor = processor

    def train_mode(self):
        FastVisionModel.for_training(self.model)

    def inference_mode(self):
        FastVisionModel.for_inference(self.model)

    def generate(self, image, use_streamer=True):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": self.user_prompt}, {"type": "image"}],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        # greedy decoding
        if use_streamer:
            text_streamer = TextStreamer(self.processor, skip_prompt=True)
            self.model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=128,
                use_cache=True,
                do_sample=False,
                )
            return ""  # Streamer prints, so return empty string or None
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                do_sample=False,
                )
            # Decode only the newly generated tokens (excluding the prompt)
            decoded_output = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return decoded_output
