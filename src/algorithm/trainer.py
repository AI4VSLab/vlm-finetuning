import logging
import wandb
from unsloth.trainer import UnslothVisionDataCollator
from torch.utils.data import DataLoader, WeightedRandomSampler
from trl import SFTTrainer, SFTConfig
from data import ChatWrapper

logger = logging.getLogger(__name__)


class BalancedSFTTrainer(SFTTrainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use a `WeightedRandomSampler` to ensure that each dataset is sampled proportionally
        to its size.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Get weights from our custom data class
        weights = self.dataset_adapter.get_sampler_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(self.train_dataset), replacement=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )



class Trainer:
    def __init__(self, data, model, config):
        self.config = config
        self.model = model
        self.data = data

        self.model.train_mode()

        model = self.model.model
        processor = self.model.processor

        # log every 1%
        num_batches = self.config.num_epochs * len(self.data.train_data)
        effective_batch_size = self.config.batch_size * self.config.grad_accum_steps
        total_steps = num_batches // effective_batch_size
        logging_steps = 1 if total_steps < 100 else total_steps // 100

        self.trainer = BalancedSFTTrainer(
            model=model,
            train_dataset=ChatWrapper(self.data.train_data, self.model.system_prompt),
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
                learning_rate=self.config.learning_rate,
                logging_steps=logging_steps,
                save_strategy="steps",
                optim="adamw_torch_fused",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,
                output_dir=f"outputs/{wandb.run.name}",
                run_name=wandb.run.name,
                report_to="wandb",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=True,
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
