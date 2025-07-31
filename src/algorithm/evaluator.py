import logging
import random
from tqdm import tqdm
from PIL import Image as PILImage
import torch
import numpy as np
from torchvision import transforms

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data # OCTExpertData for training samples
        self.test_data = test_data   # OCTTestData for test samples
        self.device = self.model.device # Get device from the model

        # Set model to inference mode
        self.model.inference_mode()

        logger.info("Evaluator initialized for sample generation.")

    def generate_train_samples(self, num_samples=5):
        """Generates and prints predictions for a small sample of images from the training dataset."""
        logger.info(f"Generating {num_samples} sample images from the training dataset...")

        if len(self.train_data.tfrecord_dataset) == 0:
            logger.warning("Training dataset is empty. Cannot generate sample predictions.")
            return

        # Ensure we can randomly access samples
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

                # Ensure image is PIL Image for model.generate
                if not isinstance(image, PILImage.Image):
                    if isinstance(image, torch.Tensor):
                        image = transforms.ToPILImage()(image.cpu())
                    elif isinstance(image, np.ndarray):
                        image = PILImage.fromarray(image.astype(np.uint8))
                    else:
                        logger.error(f"Unsupported image type for train sample {idx}: {type(image)}. Skipping.")
                        continue

                self.model.generate(image, use_streamer=True)

            except Exception as e:
                logger.error(f"Error generating train sample prediction for index {idx}: {e}")
                continue

        logger.info(f"Sample prediction generation completed for training dataset.")


    def generate_test_samples(self, num_samples=5):
        """Generates and prints predictions for a small sample of images from the test dataset."""
        logger.info(f"Generating {num_samples} sample images from the test dataset...")

        if len(self.test_data.dataset) == 0:
            logger.warning("Test dataset is empty. Cannot generate sample predictions.")
            return

        # Ensure we can randomly access samples
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

                # Ensure image is PIL Image for model.generate
                if not isinstance(image, PILImage.Image):
                    if isinstance(image, torch.Tensor):
                        image = transforms.ToPILImage()(image.cpu())
                    elif isinstance(image, np.ndarray):
                        image = PILImage.fromarray(image.astype(np.uint8))
                    else:
                        logger.error(f"Unsupported image type for test sample {idx}: {type(image)}. Skipping.")
                        continue

                self.model.generate(image, use_streamer=True)

            except Exception as e:
                logger.error(f"Error generating test sample prediction for index {idx}: {e}")
                continue

        logger.info("Sample prediction generation completed for test dataset.")

    def run(self):
        """Runs all sample generation steps."""
        logger.info("Starting sample generation process...")

        self.generate_train_samples(num_samples=5)
        self.generate_test_samples(num_samples=5)

        logger.info("Sample generation process completed.")
