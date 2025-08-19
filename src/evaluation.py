from algorithm import Evaluator
from model import VisionLanguageModel
from data import OCTTranscriptData

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


@hydra.main(version_base=None, config_path="config", config_name="evaluate")
def main(config: DictConfig) -> None:
    """Main function to run the training pipeline."""
    # --- SETUP ---
    setup()

    # --- DATA ---
    # For training, use OCTExpertData
    data = OCTTranscriptData(config.data)
    print(data.train_data[0])
    print(data.test_data[0])

    # --- MODEL ---
    model = VisionLanguageModel(config.model)

    # --- ALGORITHM ---
    evaluator = Evaluator(model, data, config.evaluator)
    evaluator.run()

    # --- CLEANUP ---
    cleanup()


if __name__ == "__main__":
    main()
