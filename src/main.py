from algorithm import Trainer
from algorithm import Evaluator
from model import VisionLanguageModel
from data import OCTTranscriptData, OCTTestData

import logging
import torch
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def setup():
    """Set up the environment."""
    torch.manual_seed(42)


def cleanup():
    pass


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    """Main function to run the training pipeline."""
    # --- SETUP ---
    setup()

    # --- DATA ---
    # For training, use OCTExpertData
    train_data = OCTTranscriptData(config.model.prompt, config.data)
    # For testing and sample generation, use OCTTestData
    test_data = OCTTestData(config.data)

    # --- MODEL ---
    model = VisionLanguageModel(config.model)

    # --- ALGORITHM ---
    trainer = Trainer(train_data, model, config.trainer)
    trainer.run()

    evaluator = Evaluator(model, train_data, test_data, config.evaluator)
    evaluator.run()

    # --- CLEANUP ---
    cleanup()


if __name__ == "__main__":
    main()
