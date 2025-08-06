import logging
import torch.nn.functional as F
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)


class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass through the model
        outputs = model(**inputs)
        # The SFTTrainer typically handles shifting of labels for language modeling
        # and computes loss based on the prediction of the next token.
        # For a standard LLM finetuning, the loss is cross-entropy.
        # logits: (batch_size, sequence_length, vocab_size)
        # labels: (batch_size, sequence_length)

        # Get logits and labels from the model output
        logits = outputs.logits
        labels = inputs["labels"]

        # Shift so that tokens < n predict n
        # This aligns the logits with the labels, as the model predicts the *next* token.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens and apply cross-entropy loss
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class Trainer:
    def __init__(self, data, model, config):
        self.config = config
        self.model = model
        self.data = data

        self.model.train_mode()

        model = self.model.model
        processor = self.model.processor
        self.trainer = CustomSFTTrainer( # Use CustomSFTTrainer here
            model=model,
            train_dataset=self.data.dataset,
            processing_class=processor.tokenizer,
            data_collator=UnslothVisionDataCollator(model, processor),
            args = SFTConfig(
                per_device_train_batch_size = self.config.batch_size,
                gradient_accumulation_steps = self.config.grad_accum_steps,
                gradient_checkpointing = True,

                # use reentrant checkpointing
                gradient_checkpointing_kwargs = {"use_reentrant": False},
                max_grad_norm = 0.3,              # max gradient norm based on QLoRA paper
                warmup_ratio = 0.03,
                #max_steps = 30,
                num_train_epochs = self.config.num_epochs,          # Set this instead of max_steps for full training runs
                learning_rate = 2e-4,
                logging_steps = 1,
                save_strategy="steps",
                optim = "adamw_torch_fused",
                weight_decay = 0.01,
                lr_scheduler_type = "cosine",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none",             # For Weights and Biases

                # You MUST put the below items for vision finetuning:
                remove_unused_columns = False,
                dataset_text_field = "",
                dataset_kwargs = {"skip_prepare_dataset": True},
                max_length = 2048,
            )
        )

    def run(self):
        logger.info("Starting training...")
        self.trainer.train()
        logger.info("Training completed.")
