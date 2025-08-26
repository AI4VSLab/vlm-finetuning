import logging
import wandb
from unsloth.trainer import UnslothVisionDataCollator
from torch.utils.data import DataLoader, WeightedRandomSampler
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from data import ChatWrapper

logger = logging.getLogger(__name__)


class BalancedSFTTrainer(SFTTrainer):
    def __init__(self, *args, sampler_weights=None, **kwargs):
        # Store the weights before calling the parent constructor
        self.sampler_weights = sampler_weights
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use a `WeightedRandomSampler` to ensure that each dataset is sampled
        in a balanced way.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # If no weights are provided, default to the standard dataloader
        if self.sampler_weights is None:
            return super().get_train_dataloader()

        # Create the sampler
        sampler = WeightedRandomSampler(
            weights=self.sampler_weights,
            num_samples=len(self.train_dataset),
            replacement=True,
        )

        # Create the DataLoader with the custom sampler
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

        sampler_weights = self.data.get_sampler_weights()
        model = self.model.model
        processor = self.model.processor

        # log every 1%
        # num_batches = self.config.num_epochs * len(self.data.train_data)
        # effective_batch_size = self.config.train_batch_size * self.config.grad_accum_steps
        # total_steps = num_batches // effective_batch_size
        # logging_steps = 1 if total_steps < 100 else total_steps // 100
        logging_steps = self.config.logging_steps
        logger.info(f"Logging Steps {logging_steps}")

        self.trainer = BalancedSFTTrainer(
            model=model,
            train_dataset=ChatWrapper(self.data.train_data, self.model.system_prompt),
            eval_dataset=ChatWrapper(self.data.val_data, self.model.system_prompt),
            sampler_weights=sampler_weights,
            processing_class=processor.tokenizer,
            data_collator=UnslothVisionDataCollator(model, processor),
            args=SFTConfig(
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                gradient_accumulation_steps=self.config.grad_accum_steps,
                gradient_checkpointing=True,
                # use reentrant checkpointing
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
                warmup_ratio=self.config.warmup_ratio,
                num_train_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                logging_steps=logging_steps,
                save_strategy="steps",
                save_steps=logging_steps,
                eval_strategy="steps",
                eval_steps=logging_steps,
                save_total_limit=3,
                load_best_model_at_end=True,  # MUST USE for early stopping
                metric_for_best_model="eval_loss",  # metric we want to early stop on
                greater_is_better=False,
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

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,  # How many steps we will wait if the eval loss doesn't decrease
            # For example the loss might increase, but decrease after 3 steps
            early_stopping_threshold=0.0,  # Can set higher - sets how much loss should decrease by until
            # we consider early stopping. For eg 0.01 means if loss was
            # 0.02 then 0.01, we consider to early stop the run.
        )
        self.trainer.add_callback(early_stopping_callback)

    def run(self):
        logger.info("Starting training...")
        self.trainer.train()
        self.model.save()
        logger.info("Training completed.")
