import logging
import random
from tqdm import tqdm
import evaluate
import nltk

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, data, config):
        self.config = config
        self.model = model
        self.data = data
        self.train_data = data.train_data  # OCTTrainData for training samples
        self.test_data = data.test_data  # OCTTestData for test samples

        logger.info("Loading text generation metrics (BLEU, METEOR, ROUGE)...")
        try:
            # METEOR requires the 'punkt' tokenizer from NLTK
            nltk.download("punkt", quiet=True)
        except Exception as e:
            logger.warning(
                f"NLTK download for 'punkt' failed: {e}. METEOR score might not work."
            )

        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        self.rouge = evaluate.load("rouge")

        # Set model to inference mode
        self.model.inference_mode()
        logger.info("Evaluator initialized for evaluation and sample generation.")

    def evaluate_train_dataset(self, num_samples=50):
        """Evaluating generates sample outputs for the training dataset."""

        all_train_samples = self.train_data.samples
        sample_indices = random.sample(
            range(len(all_train_samples)), min(num_samples, len(all_train_samples))
        )

        all_predicted_texts = []
        all_reference_texts = []

        for i, idx in enumerate(tqdm(sample_indices, desc="Evaluating Train Data")):
            sample = all_train_samples[idx]
            image = sample["image"]
            reference = sample["text"]

            # Generate full text output for parsing the diagnosis
            predicted = self.model.generate(
                image, use_streamer=False
            )  # Get full string output

            all_predicted_texts.append(predicted)
            all_reference_texts.append(reference)

        bleu_score = self.bleu.compute(
            predictions=all_predicted_texts, references=all_reference_texts
        )
        meteor_score = self.meteor.compute(
            predictions=all_predicted_texts, references=all_reference_texts
        )
        rouge_scores = self.rouge.compute(
            predictions=all_predicted_texts, references=all_reference_texts
        )

        logger.info(f"Train Dataset BLEU Score: {bleu_score['bleu']:.4f}")
        logger.info(f"Train Dataset METEOR Score: {meteor_score['meteor']:.4f}")
        logger.info(f"Train Dataset ROUGE-L Score: {rouge_scores['rougeL']:.4f}")

    def generate_train_samples(self, num_samples=5):
        """Generates and prints predictions for a small sample of images from the training dataset."""
        logger.info(
            f"Generating {num_samples} sample images from the training dataset..."
        )

        all_train_samples = self.train_data.samples
        sample_indices = random.sample(
            range(len(all_train_samples)), min(num_samples, len(all_train_samples))
        )

        for i, idx in enumerate(sample_indices):
            try:
                sample = all_train_samples[idx]
                image = sample["image"]
                true_text = sample["text"]  # The original decoded text explanation

                if image is None:
                    logger.warning(f"Skipping train sample {idx} due to missing image.")
                    continue

                logger.info(
                    f"\n--- Train Sample {i + 1}/{num_samples} (Original Index: {idx}) ---"
                )

                logger.info(f"True Features: {true_text}")
                logger.info("Model Generated Output:")

                self.model.generate(
                    image, use_streamer=True
                )  # Use streamer for sample generation

            except Exception as e:
                logger.error(
                    f"Error generating train sample prediction for index {idx}: {e}"
                )
                continue

        logger.info("Sample prediction generation completed for training dataset.")

    def generate_test_samples(self, num_samples=5):
        """Generates and prints predictions for a small sample of images from the test dataset."""
        logger.info(f"Generating {num_samples} sample images from the test dataset...")

        all_test_samples = list(self.test_data)
        sample_indices = random.sample(
            range(len(all_test_samples)), min(num_samples, len(all_test_samples))
        )

        for i, idx in enumerate(sample_indices):
            try:
                image_from_dataset, label_idx = all_test_samples[idx]
                image = image_from_dataset
                true_label_name = self.test_data.classes[label_idx]

                if image is None:
                    logger.warning(f"Skipping test sample {idx} due to missing image.")
                    continue

                logger.info(
                    f"\n--- Test Sample {i + 1}/{num_samples} (Original Index: {idx}) ---"
                )
                logger.info(f"True Label: {true_label_name}")
                logger.info("Model Generated Output:")

                self.model.generate(
                    image, use_streamer=True
                )  # Use streamer for sample generation

            except Exception as e:
                logger.error(
                    f"Error generating test sample prediction for index {idx}: {e}"
                )
                continue

        logger.info("Sample prediction generation completed for test dataset.")

    def run(self):
        """Runs all evaluation and sample generation steps."""
        logger.info("Starting evaluation and sample generation process...")

        eval_samples = self.config.eval_samples
        gen_samples = self.config.gen_samples

        self.evaluate_train_dataset(num_samples=eval_samples)
        self.generate_train_samples(num_samples=gen_samples)

        self.generate_test_samples(num_samples=gen_samples)
        logger.info("Test dataset evaluation and sample generation completed.")

        logger.info("Evaluation completed.")
