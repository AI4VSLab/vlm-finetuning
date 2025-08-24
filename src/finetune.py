from algorithm import Trainer
from algorithm import Evaluator
from model import VisionLanguageModel
from data import OCTBiomarkerData

import logging
import wandb
import torch
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def setup():
    """Set up the environment."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def cleanup():
    if wandb.run:
        wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(config: DictConfig) -> None:
    """Main function to run the training pipeline."""
    # --- SETUP ---
    setup()

    # --- WANDB ---
    wandb.init(
        entity="AI4VS-Lab",
        project="medgemma-finetuning",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )

    # --- DATA ---
    # For training, use OCTExpertData
    data = OCTBiomarkerData(config.data)
    print(data.train_data[0])
    print(data.test_data[0])

    # --- MODEL ---
    model = VisionLanguageModel(config.model)

    # --- ALGORITHM ---
    trainer = Trainer(data, model, config.trainer)
    trainer.run()

    evaluator = Evaluator(model, data, config.evaluator)
    evaluator.run()

    # --- CLEANUP ---
    cleanup()


if __name__ == "__main__":
    main()
