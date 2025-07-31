from algorithm import Trainer
from algorithm import Evaluator
from model import VisionLanguageModel
from data import OCTExpertData, OCTTestData

import logging
import torch
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def setup(config):
    """Set up the environment."""
    torch.manual_seed(42)
    device = torch.device(config.device)
    logger.info(f"Device: {device}")
    return device


def cleanup():
    pass


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    """Main function to run the training pipeline."""
    # --- SETUP ---
    device = setup(config.system)

    # --- DATA ---
    # For training, use OCTExpertData
    train_data = OCTExpertData(config.data)
    # For testing and sample generation, use OCTTestData
    test_data = OCTTestData(config.data)

    # --- MODEL ---
    model = VisionLanguageModel(config.model, device)

    # --- ALGORITHM ---
    trainer = Trainer(train_data, model, config.trainer)
    trainer.run()

    evaluator = Evaluator(model, train_data, test_data, config.evaluator)
    evaluator.run()

    # --- CLEANUP ---
    cleanup()


if __name__ == "__main__":
    main()
