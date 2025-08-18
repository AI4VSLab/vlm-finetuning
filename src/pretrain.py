from algorithm import Trainer, Evaluator
from model import VisionLanguageModel
from data import OCTDiagnosisData

import logging
import torch
import random
import numpy as np
import hydra
import wandb
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


@hydra.main(version_base=None, config_path="config", config_name="pretrain")
def main(config: DictConfig) -> None:
    """Main function to run the training pipeline."""
    # --- SETUP ---
    setup()

    # --- WANDB ---
    wandb.init(
        entity='AI4VS-Lab',
        project='medgemma-pretraining',
        config=OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
    )

    # --- DATA ---
    # For training, use OCTExpertData
    data = OCTDiagnosisData(config.data)
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
