import logging
import random
from tqdm import tqdm
import re
import evaluate
import nltk

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, train_data, test_data, config):
        self.config = config
        self.model = model
        self.train_data = train_data  # OCTTrainData for training samples
        self.test_data = test_data  # OCTTestData for test samples

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

    def _extract_output(self, generated_text: str) -> tuple[str | None, str | None]:
        """
        Extracts the explanation and diagnosis from the model's generated output.

        Args:
            generated_text: The raw output from the language model.

        Returns:
            A tuple containing (explanation, diagnosis). Each element is None if not found.
        """
        if not isinstance(generated_text, str):
            logger.warning(f"Generated text is not a string: {type(generated_text)}")
            return None, None

        # Explanation extraction
        exp_match = re.search(
            r"<explanation>(.*?)</explanation>",
            generated_text,
            re.IGNORECASE | re.DOTALL,
        )
        explanation = exp_match.group(1).strip() if exp_match else None

        # Diagnosis extraction
        diag_match = re.search(
            r"<diagnosis>\s*(AMD|Normal)</diagnosis>", generated_text, re.IGNORECASE
        )
        diagnosis = diag_match.group(1).capitalize() if diag_match else None

        if not explanation:
            logger.warning(
                f'Could not extract explanation from: "{generated_text.strip()[:100]}..."'
            )
        if not diagnosis:
            logger.warning(
                f'Could not extract diagnosis from: "{generated_text.strip()[:100]}..."'
            )

        return explanation, diagnosis

    def evaluate_train_dataset(self, num_samples=50):
        """Evaluates classification accuracy and generates sample outputs for the training dataset."""
        logger.info(
            f"Evaluating classification accuracy on {num_samples} samples from the training dataset..."
        )

        all_train_samples = list(self.train_data.tfrecord_dataset)
        sample_indices_for_accuracy = random.sample(
            range(len(all_train_samples)), min(num_samples, len(all_train_samples))
        )

        correct_predictions = 0
        total_predictions = 0

        all_predicted_texts = []
        all_reference_texts = []

        for i, idx in enumerate(
            tqdm(sample_indices_for_accuracy, desc="Evaluating Train Data Accuracy")
        ):
            try:
                sample = all_train_samples[idx]
                image = sample["image"]
                true_diagnosis_str = "Normal" if sample["label"] == 0 else "AMD"
                reference_explanation = sample["text"]

                # Generate full text output for parsing the diagnosis
                generated_output = self.model.generate(
                    image, use_streamer=False
                )  # Get full string output
                predicted_explanation, predicted_diagnosis = self._extract_output(
                    generated_output
                )

                if predicted_explanation is None or predicted_diagnosis is None:
                    logger.warning(
                        f"Could not extract for train sample {idx}. Skipping... Generated output: {generated_output[:100]}..."
                    )
                    continue

                if (
                    predicted_diagnosis.upper() == true_diagnosis_str.upper()
                ):  # Case-insensitive comparison
                    correct_predictions += 1
                total_predictions += 1

                all_predicted_texts.append(predicted_explanation)
                all_reference_texts.append(reference_explanation)

            except Exception as e:
                logger.error(
                    f"Error evaluating train sample for accuracy at index {idx}: {e}"
                )
                continue

        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            bleu_score = self.bleu.compute(
                predictions=all_predicted_texts, references=all_reference_texts
            )
            meteor_score = self.meteor.compute(
                predictions=all_predicted_texts, references=all_reference_texts
            )
            rouge_scores = self.rouge.compute(
                predictions=all_predicted_texts, references=all_reference_texts
            )

            logger.info(
                f"Train Dataset Classification Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})"
            )
            logger.info(f"Train Dataset BLEU Score: {bleu_score['bleu']:.4f}")
            logger.info(f"Train Dataset METEOR Score: {meteor_score['meteor']:.4f}")
            logger.info(f"Train Dataset ROUGE-L Score: {rouge_scores['rougeL']:.4f}")
        else:
            logger.info("No samples to calculate metrics.")

    def evaluate_test_dataset(self, num_samples=50):
        """Evaluates classification accuracy and generates sample outputs for the test dataset."""
        logger.info(
            f"Evaluating classification accuracy on {num_samples} samples from the test dataset..."
        )

        all_test_samples = list(self.test_data.dataset)

        sample_indices_for_accuracy = random.sample(
            range(len(all_test_samples)), min(num_samples, len(all_test_samples))
        )

        correct_predictions = 0
        total_predictions = 0

        for i, idx in enumerate(
            tqdm(sample_indices_for_accuracy, desc="Evaluating Test Data Accuracy")
        ):
            try:
                image_from_dataset, label_idx = all_test_samples[idx]
                image = image_from_dataset
                true_label_name = "Normal" if label_idx == 0 else "AMD"

                if image is None:
                    logger.warning(
                        f"Skipping test sample {idx} for accuracy due to missing image."
                    )
                    continue

                # Generate full text output for parsing the diagnosis
                generated_output = self.model.generate(
                    image, use_streamer=False
                )  # Get full string output
                _, predicted_diagnosis = self._extract_output(generated_output)

                if predicted_diagnosis is None:
                    logger.warning(
                        f"Could not extract diagnosis for test sample {idx}. Skipping for accuracy. Generated output: {generated_output[:100]}..."
                    )
                    continue

                if (
                    predicted_diagnosis.upper() == true_label_name.upper()
                ):  # Case-insensitive comparison
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                logger.error(
                    f"Error evaluating test sample for accuracy at index {idx}: {e}"
                )
                continue

        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            logger.info(
                f"Test Dataset Classification Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})"
            )
        else:
            logger.info("No valid test samples processed for accuracy calculation.")

    def generate_train_samples(self, num_samples=5):
        """Generates and prints predictions for a small sample of images from the training dataset."""
        logger.info(
            f"Generating {num_samples} sample images from the training dataset..."
        )

        all_train_samples = list(self.train_data.tfrecord_dataset)
        sample_indices = random.sample(
            range(len(all_train_samples)), min(num_samples, len(all_train_samples))
        )

        for i, idx in enumerate(sample_indices):
            try:
                sample = all_train_samples[idx]
                image = sample["image"]
                true_diagnosis_str = "Normal" if sample["label"] == 0 else "AMD"
                true_text = sample["text"]  # The original decoded text explanation

                if image is None:
                    logger.warning(f"Skipping train sample {idx} due to missing image.")
                    continue

                logger.info(
                    f"\n--- Train Sample {i + 1}/{num_samples} (Original Index: {idx}) ---"
                )
                logger.info(f"True Diagnosis: {true_diagnosis_str}")
                logger.info(f"True Explanation: {true_text}")
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

        all_test_samples = list(self.test_data.dataset)
        sample_indices = random.sample(
            range(len(all_test_samples)), min(num_samples, len(all_test_samples))
        )

        for i, idx in enumerate(sample_indices):
            try:
                image_from_dataset, label_idx = all_test_samples[idx]
                image = image_from_dataset
                true_label_name = self.test_data.dataset.classes[label_idx]

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

        num_accuracy_samples = self.config.num_samples
        num_generation_samples = 5

        self.evaluate_train_dataset(num_samples=num_accuracy_samples)
        self.generate_train_samples(num_samples=num_generation_samples)
        # Generate samples with text streamer after accuracy evaluation

        self.evaluate_test_dataset(num_samples=num_accuracy_samples)
        self.generate_test_samples(num_samples=num_generation_samples)
        logger.info("Test dataset evaluation and sample generation completed.")

        logger.info("Evaluation completed.")
