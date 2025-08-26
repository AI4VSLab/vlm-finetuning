# amd-vlm-finetuning/src/evaluate.py
import logging
import torch
import random
import numpy as np
import hydra
from omegaconf import DictConfig

from algorithm import Evaluator
from model import VisionLanguageModel
from data import OCTEvaluationData

logger = logging.getLogger(__name__)


def setup():
    """Set up the environment for reproducibility."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="config", config_name="evaluate")
def main(config: DictConfig) -> None:
    """Main function to run the evaluation pipeline."""
    # --- SETUP ---
    setup()

    # --- DATA ---
    logger.info("Loading evaluation data...")
    # OCTEvaluationData loads the 'test' splits of your datasets
    data = OCTEvaluationData(config.data)

    # --- MODEL ---
    logger.info("Loading fine-tuned model for evaluation...")
    # The model config now points to your checkpoint.
    # The VisionLanguageModel class will handle loading the base model,
    # processor, and applying the saved adapter weights.
    model = VisionLanguageModel(config.model)

    # --- EVALUATOR ---
    logger.info("Initializing evaluator...")
    evaluator = Evaluator(model, data, config.evaluator)

    # --- RUN EVALUATION ---
    logger.info("Starting evaluation...")
    evaluator.run()
    logger.info("Evaluation script finished.")


if __name__ == "__main__":
    main()
