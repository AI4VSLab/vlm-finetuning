import logging
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, data, model, config):
        self.config = config
        self.model = model
        self.data = data

        self.model.train_mode()

        model = self.model.model
        processor = self.model.processor

        self.trainer = SFTTrainer(
            model=model,
            train_dataset=self.data.train_data,
            processing_class=processor.tokenizer,
            data_collator=UnslothVisionDataCollator(model, processor),
            args=SFTConfig(
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.grad_accum_steps,
                gradient_checkpointing=True,
                # use reentrant checkpointing
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
                warmup_ratio=0.03,
                num_train_epochs=self.config.num_epochs,
                # max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                logging_steps=self.config.logging_steps,
                save_strategy="steps",
                optim="adamw_torch_fused",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,
                output_dir="outputs",
                report_to="none",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=2048,
            ),
        )

    def run(self):
        logger.info("Starting training...")
        self.trainer.train()
        self.model.save()
        logger.info("Training completed.")
