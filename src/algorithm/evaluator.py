import logging
import random
from tqdm import tqdm
from PIL import Image as PILImage
import evaluate

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


    def evaluate_train_dataset(self, num_samples=50):
        """Evaluates classification accuracy and generates sample outputs for the training dataset."""
        logger.info(f"Evaluating classification accuracy on {num_samples} samples from the training dataset...")

        all_train_samples = list(self.train_data.tfrecord_dataset)
        sample_indices = random.sample(range(len(all_train_samples)), min(num_samples, len(all_train_samples)))

        predictions = []
        references = []

        for i, idx in enumerate(tqdm(sample_indices_for_accuracy, desc="Evaluating Train Data Accuracy")):
            sample = all_train_samples[idx]
            image = sample["image"]

            ground_truth = sample["text"]
            references.append(ground_truth)

            # Generate full text output for parsing the diagnosis
            generated_output = self.model.generate(image, use_streamer=False) # Get full string output
            predictions.append(generated_output)

        print(references)
        print(predictions)

    def generate_train_samples(self, num_samples=5):
        """Generates and prints predictions for a small sample of images from the training dataset."""
        logger.info(f"Generating {num_samples} sample images from the training dataset...")

        all_train_samples = list(self.train_data.tfrecord_dataset)
        sample_indices = random.sample(range(len(all_train_samples)), min(num_samples, len(all_train_samples)))

        for i, idx in enumerate(sample_indices):
            try:
                sample = all_train_samples[idx]
                image = sample["image"]
                true_text = sample["text"] # The original decoded text explanation

                if image is None:
                    logger.warning(f"Skipping train sample {idx} due to missing image.")
                    continue

                logger.info(f"\n--- Train Sample {i+1}/{num_samples} (Original Index: {idx}) ---")
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

        # Generate samples with text streamer after accuracy evaluation

        self.generate_train_samples(num_samples=num_generation_samples)
        self.generate_test_samples(num_samples=num_generation_samples)
        logger.info("Test dataset evaluation and sample generation completed.")

        logger.info("Evaluation completed.")
