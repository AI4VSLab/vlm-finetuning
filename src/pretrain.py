from algorithm import Trainer
from algorithm import Evaluator
from model import VisionLanguageModel
from data import OCTDiagnosisData

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


@hydra.main(version_base=None, config_path="config", config_name="pretrain")
def main(config: DictConfig) -> None:
    """Main function to run the training pipeline."""
    # --- SETUP ---
    setup()

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
