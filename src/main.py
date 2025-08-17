from algorithm import Trainer
from algorithm import Evaluator
from model import VisionLanguageModel
from data import OCTTranscriptData

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
    data = OCTTranscriptData(config)

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
