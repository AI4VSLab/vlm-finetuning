import logging
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, data, model, config):
        self.config = config
        self.model = model
        self.data = data

        self.model.train_mode()


        args = SFTConfig(
            output_dir="gemma-product-description",     # directory to save and repository id
            num_train_epochs=self.config.num_epochs,                         # number of training epochs
            per_device_train_batch_size=self.config.batch_size,              # batch size per device during training
            gradient_accumulation_steps=self.config.grad_accum_steps,              # number of steps before performing a backward/update pass
            gradient_checkpointing=True,                # use gradient checkpointing to save memory
            optim="adamw_torch_fused",                  # use fused adamw optimizer
            logging_steps=1,                            # log every 5 steps
            save_strategy="epoch",                      # save checkpoint every epoch
            learning_rate=2e-4,                         # learning rate, based on QLoRA paper
            bf16=True,                                  # use bfloat16 precision
            max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
            warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
            lr_scheduler_type="constant",               # use constant learning rate scheduler
            push_to_hub=False,                           # push model to hub                   # report metrics to tensorboard
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },  # use reentrant checkpointing
            dataset_text_field="",                      # need a dummy field for collator
            dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
        )
        args.remove_unused_columns = False # important for collator


        self.trainer = SFTTrainer(
            model=self.model.model,
            args=args,
            train_dataset=self.data.dataset,
            processing_class=self.model.processor,
            data_collator=self.model.data_collator,
        )

    def run(self):
        logger.info("Starting training...")
        self.trainer.train()
        logger.info("Training completed.")
        self.trainer.save_model()
        logger.info("Model Saved.")
