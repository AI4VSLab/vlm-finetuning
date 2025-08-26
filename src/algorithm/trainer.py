import logging
import wandb
from unsloth.trainer import UnslothVisionDataCollator
from torch.utils.data import DataLoader, WeightedRandomSampler
from trl import SFTTrainer, SFTConfig
# Import TrainerCallback to create our custom save function
from transformers import EarlyStoppingCallback, TrainerCallback
from data import ChatWrapper
import os

logger = logging.getLogger(__name__)


# --- 1. Create the Custom Save Function (Callback) ---
# This class will save the processor every time a model checkpoint is saved.
class SaveProcessorCallback(TrainerCallback):
    """A custom callback to save the processor with each model checkpoint."""

    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        """
        This event is triggered after a checkpoint is saved.
        """
        # The path to the checkpoint directory (e.g., "outputs/run_name/checkpoint-500")
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # Save the processor to the same directory as the model checkpoint
        self.processor.save_pretrained(checkpoint_path)


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

        if self.sampler_weights is None:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=self.sampler_weights,
            num_samples=len(self.train_dataset),
            replacement=True
        )

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
        hf_model = self.model.model
        processor = self.model.processor

        logging_steps = self.config.logging_steps
        logger.info(f"Logging Steps {logging_steps}")

        self.trainer = BalancedSFTTrainer(
            model=hf_model,
            train_dataset=ChatWrapper(self.data.train_data, self.model.system_prompt),
            eval_dataset=ChatWrapper(self.data.val_data, self.model.system_prompt),
            sampler_weights=sampler_weights,
            processing_class=processor.tokenizer,
            data_collator=UnslothVisionDataCollator(hf_model, processor),
            args=SFTConfig(
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                gradient_accumulation_steps=self.config.grad_accum_steps,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_grad_norm=0.3,
                warmup_ratio=self.config.warmup_ratio,
                num_train_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                logging_steps=logging_steps,
                save_strategy="steps",
                save_steps=logging_steps,
                eval_strategy="steps",
                eval_steps=logging_steps,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                optim="adamw_torch_fused",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,
                output_dir=f"outputs/{wandb.run.name}",
                run_name=wandb.run.name,
                report_to="wandb",
                remove_unused_columns=True,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=2048,
            ),
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.0,
        )
        self.trainer.add_callback(early_stopping_callback)

        # --- 2. Add the custom save callback to the trainer ---
        save_processor_callback = SaveProcessorCallback(processor=processor)
        self.trainer.add_callback(save_processor_callback)

    def run(self):
        logger.info("Starting training...")
        self.trainer.train()

        # --- 3. Explicitly save the final best model and processor ---
        # After training, `load_best_model_at_end` ensures the best model is loaded.
        # We now save that final version to the main output directory.
        logger.info("Training completed. Saving final model and processor...")
        final_output_dir = self.trainer.args.output_dir
        self.trainer.save_model(final_output_dir)
        self.model.processor.save_pretrained(final_output_dir)
        logger.info(f"Final model and processor saved to {final_output_dir}")
