from unsloth import FastVisionModel, get_chat_template
from transformers import TextStreamer
import logging

logger = logging.getLogger(__name__)


class VisionLanguageModel:
    def __init__(self, config, device):
        logger.info("Creating model...")
        self.config = config
        self.device = device
        model, processor = FastVisionModel.from_pretrained(
            self.config.model_name,
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

        self.instruction = """You are a highly skilled AI assistant specializing in ophthalmic diagnostics. Your task is to analyze Optical Coherence Tomography (OCT) images to determine the presence of Age-related Macular Degeneration (AMD) or if the image appears normal. Provide a concise explanation of your findings and a definitive diagnosis.
        Your output must strictly follow this format:
        Explanation: [Your detailed but concise reasoning based on OCT findings]
        Diagnosis: [AMD or Normal]
        """

        processor = get_chat_template(
            processor,
            "gemma-3"
        )

        self.model = model
        self.processor = processor
        logger.info("Model created.")

    def generate(self, image):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction}, {"type": "image"}],
            }
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        text_streamer = TextStreamer(self.processor, skip_prompt=True)
        result = self.model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                                use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)
        return result
