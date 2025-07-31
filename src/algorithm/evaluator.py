import logging
import re
import random
from tqdm import tqdm
from PIL import Image as PILImage
import torch
import numpy as np # Added for numpy array handling
from torchvision import transforms

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data # OCTExpertData for full evaluation
        self.test_data = test_data   # OCTTestData for sample generation
        self.device = self.model.device # Get device from the model

        # Set model to inference mode
        self.model.inference_mode()

        # Regex to extract diagnosis from model output
        self.diagnosis_pattern = re.compile(r"Diagnosis: (AMD|Normal)")
        logger.info("Evaluator initialized.")

    def _parse_diagnosis(self, text):
        """Parses the diagnosis from the generated text."""
        match = self.diagnosis_pattern.search(text)
        if match:
            return match.group(1)
        return "N/A" # Not available or not parsed

    def evaluate_train_dataset(self):
        """Calculates the accuracy of the model on the entire OCTExpertData (training) dataset and prints samples."""
        logger.info("Starting training dataset evaluation...")
        correct_predictions = 0
        total_predictions = 0

        # Store samples for later generation
        all_train_samples = list(self.train_data.tfrecord_dataset)

        # Iterate through the expert_data.tfrecord_dataset for accuracy calculation
        for i, sample in enumerate(tqdm(all_train_samples, desc="Evaluating training dataset")):
            try:
                image = sample["image"]
                true_diagnosis = "Normal" if sample["label"] == 0 else "AMD"

                if image is None:
                    logger.warning(f"Skipping sample {i} due to missing image.")
                    continue

                generated_text = self.model.generate(image, use_streamer=False)
                print(generated_text)
                predicted_diagnosis = self._parse_diagnosis(generated_text)

                logger.debug(f"Sample {i}: True='{true_diagnosis}', Predicted='{predicted_diagnosis}'")

                if predicted_diagnosis == true_diagnosis:
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue

        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        logger.info(f"Training dataset evaluation completed. Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

        return accuracy

    def evaluate_test_dataset(self):
        """Calculates the accuracy of the model on the entire OCTTestData (test) dataset and prints samples."""
        logger.info("Starting test dataset evaluation...")
        correct_predictions = 0
        total_predictions = 0

        # Store samples for later generation
        all_test_samples = list(self.test_data.dataset)

        if len(all_test_samples) == 0:
            logger.warning("Test dataset is empty. Cannot perform full evaluation.")
            return 0.0

        for i, (image, label_idx) in enumerate(tqdm(all_test_samples, desc="Evaluating test dataset")):
            try:
                true_diagnosis = self.test_data.dataset.classes[label_idx]

                if not isinstance(image, PILImage.Image):
                    if isinstance(image, torch.Tensor):
                        image = transforms.ToPILImage()(image.cpu())
                    elif isinstance(image, np.ndarray):
                        image = PILImage.fromarray(image.astype(np.uint8))
                    else:
                        logger.error(f"Unsupported image type for test sample {i}: {type(image)}. Skipping.")
                        continue

                generated_text = self.model.generate(image, use_streamer=False)
                predicted_diagnosis = self._parse_diagnosis(generated_text)

                logger.debug(f"Test Sample {i}: True='{true_diagnosis}', Predicted='{predicted_diagnosis}'")

                if predicted_diagnosis == true_diagnosis:
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                logger.error(f"Error evaluating test sample {i}: {e}")
                continue

        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        logger.info(f"Test dataset evaluation completed. Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

        return accuracy

    def run(self):
        """Runs all evaluation steps."""
        logger.info("Starting full evaluation process...")

        # Evaluate on training data
        self.evaluate_train_dataset()

        # Evaluate on test data
        self.evaluate_test_dataset()

        logger.info("Full evaluation process completed.")
