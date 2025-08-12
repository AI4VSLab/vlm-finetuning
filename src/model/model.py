from unsloth import FastVisionModel, get_chat_template
from transformers import TextStreamer
import logging

logger = logging.getLogger(__name__)


class VisionLanguageModel:
    def __init__(self, config):
        logger.info("Creating model...")
        self.config = config
        self.load()
        self.prompt = self.config.prompt
        logger.info("Model created.")

    def save(self):
        self.model.save_pretrained(self.config.path)
        self.processor.save_pretrained(self.config.path)

    def load(self):
        if self.config.from_checkpoint:
            model, processor = FastVisionModel.from_pretrained(
                    model_name=self.config.path,  # YOUR MODEL YOU USED FOR TRAINING
                    load_in_4bit=True,  # Set to False for 16bit LoRA
                )
        else:
            model, processor = FastVisionModel.from_pretrained(
                self.config.name,
                load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
            )
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers     = True, # False if not finetuning vision layers
                finetune_language_layers   = True, # False if not finetuning language layers
                finetune_attention_modules = True, # False if not finetuning attention layers
                finetune_mlp_modules       = True, # False if not finetuning MLP layers

                r = 16,                           # The larger, the higher the accuracy, but might overfit
                lora_alpha = 16,                  # Recommended alpha == r at least
                lora_dropout = 0,
                bias = "none",
                random_state = 3407,
                use_rslora = False,               # We support rank stabilized LoRA
                loftq_config = None,               # And LoftQ
                target_modules = "all-linear",    # Optional now! Can specify a list if needed
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
        ).to("cuda")

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
