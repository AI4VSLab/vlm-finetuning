import logging
import random
from tqdm import tqdm
from PIL import Image as PILImage
import torch
import numpy as np
from torchvision import transforms
import re # Import regex module

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model, train_data, test_data, config):
        self.config = config
        self.model = model
        self.train_data = train_data # OCTExpertData for training samples
        self.test_data = test_data   # OCTTestData for test samples
        self.device = self.model.device # Get device from the model

        # Set model to inference mode
        self.model.inference_mode()

        logger.info("Evaluator initialized for evaluation and sample generation.")

    def _extract_diagnosis(self, generated_text: str) -> str | None:
        """
        Extracts the diagnosis (AMD or Normal) from the generated text using regex.
        Expected format: "Diagnosis: [AMD or Normal]"
        """
        if not isinstance(generated_text, str):
            logger.warning(f"Generated text is not a string: {type(generated_text)}")
            return None

        # Regex to find "Diagnosis: " followed by "AMD" or "Normal"
        match = re.search(r"<diagnosis>\s*(AMD|Normal)</diagnosis>", generated_text, re.IGNORECASE)
        if match:
            # Return the captured group, ensuring consistent casing (e.g., "AMD" or "Normal")
            return match.group(1).capitalize()
        else:
            logger.warning(f"Could not extract diagnosis from generated text: \"{generated_text.strip()[:100]}...\"")
            return None

    def evaluate_train_dataset(self, num_samples=50):
        """Evaluates classification accuracy and generates sample outputs for the training dataset."""
        logger.info(f"Evaluating classification accuracy on {num_samples} samples from the training dataset...")

        all_train_samples = list(self.train_data.tfrecord_dataset)
        sample_indices_for_accuracy = random.sample(range(len(all_train_samples)), min(num_samples, len(all_train_samples)))

        correct_predictions = 0
        total_predictions = 0

        for i, idx in enumerate(tqdm(sample_indices_for_accuracy, desc="Evaluating Train Data Accuracy")):
            try:
                sample = all_train_samples[idx]
                image = sample["image"]
                true_diagnosis_str = "Normal" if sample["label"] == 0 else "AMD"

                # Generate full text output for parsing the diagnosis
                generated_output = self.model.generate(image, use_streamer=False) # Get full string output
                predicted_diagnosis = self._extract_diagnosis(generated_output)

                if predicted_diagnosis is None:
                    logger.warning(f"Could not extract diagnosis for train sample {idx}. Skipping for accuracy. Generated output: {generated_output[:100]}...")
                    continue

                if predicted_diagnosis.upper() == true_diagnosis_str.upper(): # Case-insensitive comparison
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                logger.error(f"Error evaluating train sample for accuracy at index {idx}: {e}")
                continue

        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            logger.info(f"Train Dataset Classification Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        else:
            logger.info("No valid train samples processed for accuracy calculation.")


    def evaluate_test_dataset(self, num_samples=50):
        """Evaluates classification accuracy and generates sample outputs for the test dataset."""
        logger.info(f"Evaluating classification accuracy on {num_samples} samples from the test dataset...")

        all_test_samples = list(self.test_data.dataset)

        sample_indices_for_accuracy = random.sample(range(len(all_test_samples)), min(num_samples, len(all_test_samples)))

        correct_predictions = 0
        total_predictions = 0

        for i, idx in enumerate(tqdm(sample_indices_for_accuracy, desc="Evaluating Test Data Accuracy")):
            try:
                image_from_dataset, label_idx = all_test_samples[idx]
                image = image_from_dataset
                true_label_name = "Normal" if label_idx == 0 else "AMD"

                if image is None:
                    logger.warning(f"Skipping test sample {idx} for accuracy due to missing image.")
                    continue

                # Generate full text output for parsing the diagnosis
                generated_output = self.model.generate(image, use_streamer=False) # Get full string output
                predicted_diagnosis = self._extract_diagnosis(generated_output)

                if predicted_diagnosis is None:
                    logger.warning(f"Could not extract diagnosis for test sample {idx}. Skipping for accuracy. Generated output: {generated_output[:100]}...")
                    continue

                if predicted_diagnosis.upper() == true_label_name.upper(): # Case-insensitive comparison
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                logger.error(f"Error evaluating test sample for accuracy at index {idx}: {e}")
                continue

        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            logger.info(f"Test Dataset Classification Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        else:
            logger.info("No valid test samples processed for accuracy calculation.")

    def generate_train_samples(self, num_samples=5):
        """Generates and prints predictions for a small sample of images from the training dataset."""
        logger.info(f"Generating {num_samples} sample images from the training dataset...")

        all_train_samples = list(self.train_data.tfrecord_dataset)
        sample_indices = random.sample(range(len(all_train_samples)), min(num_samples, len(all_train_samples)))

        for i, idx in enumerate(sample_indices):
            try:
                sample = all_train_samples[idx]
                image = sample["image"]
                true_diagnosis_str = "Normal" if sample["label"] == 0 else "AMD"
                true_text = sample["text"] # The original decoded text explanation

                if image is None:
                    logger.warning(f"Skipping train sample {idx} due to missing image.")
                    continue

                logger.info(f"\n--- Train Sample {i+1}/{num_samples} (Original Index: {idx}) ---")
                logger.info(f"True Diagnosis: {true_diagnosis_str}")
                logger.info(f"True Explanation: {true_text}")
                logger.info("Model Generated Output:")

                self.model.generate(image, use_streamer=True) # Use streamer for sample generation

            except Exception as e:
                logger.error(f"Error generating train sample prediction for index {idx}: {e}")
                continue

        logger.info("Sample prediction generation completed for training dataset.")


    def generate_test_samples(self, num_samples=5):
        """Generates and prints predictions for a small sample of images from the test dataset."""
        logger.info(f"Generating {num_samples} sample images from the test dataset...")

        all_test_samples = list(self.test_data.dataset)
        sample_indices = random.sample(range(len(all_test_samples)), min(num_samples, len(all_test_samples)))

        for i, idx in enumerate(sample_indices):
            try:
                image_from_dataset, label_idx = all_test_samples[idx]
                image = image_from_dataset
                true_label_name = self.test_data.dataset.classes[label_idx]

                if image is None:
                    logger.warning(f"Skipping test sample {idx} due to missing image.")
                    continue

                logger.info(f"\n--- Test Sample {i+1}/{num_samples} (Original Index: {idx}) ---")
                logger.info(f"True Label: {true_label_name}")
                logger.info("Model Generated Output:")

                self.model.generate(image, use_streamer=True) # Use streamer for sample generation

            except Exception as e:
                logger.error(f"Error generating test sample prediction for index {idx}: {e}")
                continue

        logger.info("Sample prediction generation completed for test dataset.")

    def run(self):
        """Runs all evaluation and sample generation steps."""
        logger.info("Starting evaluation and sample generation process...")

        num_accuracy_samples = self.config.num_samples
        num_generation_samples = 5

        self.evaluate_train_dataset(num_samples=num_accuracy_samples)
        self.generate_train_samples(num_samples=num_generation_samples)
        # Generate samples with text streamer after accuracy evaluation

        self.evaluate_test_dataset(num_samples=num_accuracy_samples)
        self.generate_test_samples(num_samples=num_generation_samples)
        logger.info("Test dataset evaluation and sample generation completed.")

        logger.info("Evaluation completed.")
