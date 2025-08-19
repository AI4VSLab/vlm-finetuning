import logging
import random
from tqdm import tqdm
import numpy as np
import evaluate
import nltk
import json
import re

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, data, config):
        self.config = config
        self.model = model
        self.data = data  # data should have train_data and test_data for evaluation

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
        self.bert = evaluate.load("bertscore")


        # Set model to inference mode
        self.model.inference_mode()
        logger.info("Evaluator initialized for evaluation and sample generation.")

    def evaluate(self, dataset, num_samples=50):
        """
        Evaluating generates sample outputs for the input dataset.
        The dataset has the following columns (image, text, label)
        """

        all_samples = dataset
        sample_indices = random.sample(
            range(len(all_samples)), min(num_samples, len(all_samples))
        )

        all_predicted_texts = []
        all_reference_texts = []
        correct_predictions, total_predictions = 0, 0

        for i, idx in enumerate(tqdm(sample_indices, desc="Evaluating Data")):
            sample = all_samples[idx]
            image = sample["image"]
            label = sample["label"]
            reference = sample["text"]

            # Generate full text output for parsing the diagnosis
            predicted = self.model.generate(
                image, use_streamer=False
            )  # Get full string output

            all_predicted_texts.append(predicted)
            all_reference_texts.append(reference)

            try:
                parsed_output = json.loads(predicted)
                predicted_diagnosis = parsed_output.get("diagnosis", " ")
                # Extract the class name from the diagnosis string, e.g., "Description (CLASS)"
                match = re.search(r"\((\w+)\)", predicted_diagnosis)
                if match:
                    predicted_label = match.group(1)
                    if predicted_label == label:
                        correct_predictions += 1
                total_predictions += 1

            except (json.JSONDecodeError, AttributeError):
                logger.warning(
                    f"Could not parse model output for accuracy: {predicted}"
                )
                continue

        bleu_score = self.bleu.compute(
            predictions=all_predicted_texts, references=all_reference_texts
        )
        meteor_score = self.meteor.compute(
            predictions=all_predicted_texts, references=all_reference_texts
        )
        rouge_scores = self.rouge.compute(
            predictions=all_predicted_texts, references=all_reference_texts
        )

        bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")

        accuracy = (
            (correct_predictions / total_predictions) * 100
            if total_predictions > 0
            else 0
        )

        logger.info(f"Classification Accuracy: {accuracy:.2f}%")
        logger.info(
            f"Lexical Metrics -  BLEU: {bleu_score['bleu']:.4f}, METEOR: {meteor_score['meteor']:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}"
        )
        logger.info(f"Semantic Metrics - BERT: {np.mean(bert_scores['f1']):.4f}")

    def generate(self, dataset, num_samples=5):
        """Generates and prints predictions for a small sample of images from the input dataset."""

        all_samples = dataset
        sample_indices = random.sample(
            range(len(all_samples)), min(num_samples, len(all_samples))
        )

        for i, idx in enumerate(sample_indices):
            try:
                sample = all_samples[idx]
                image = sample["image"]
                reference_output = sample[
                    "text"
                ]  # The original decoded text explanation

                if image is None:
                    logger.warning(f"Skipping train sample {idx} due to missing image.")
                    continue

                logger.info(
                    f"\n--- Train Sample {i + 1}/{num_samples} (Original Index: {idx}) ---"
                )

                logger.info(f"Reference Output: {reference_output}")
                logger.info("Predicted Output:")

                self.model.generate(
                    image, use_streamer=True
                )  # Use streamer for sample generation

            except Exception as e:
                logger.error(
                    f"Error generating train sample prediction for index {idx}: {e}"
                )
                continue

        logger.info("Sample prediction generation completed for training dataset.")

    def run(self):
        """Runs all evaluation and sample generation steps."""
        logger.info("Starting evaluation and sample generation process...")

        eval_samples = self.config.eval_samples
        gen_samples = self.config.gen_samples

        logger.info("Train dataset Evaluation: ")
        self.evaluate(self.data.train_data, num_samples=eval_samples)

        logger.info("Train dataset Generation: ")
        self.generate(self.data.train_data, num_samples=gen_samples)

        logger.info("Test dataset Evaluation: ")
        self.evaluate(self.data.test_data, num_samples=eval_samples)

        logger.info("Test dataset Generation: ")
        self.generate(self.data.test_data, num_samples=gen_samples)

        logger.info("Evaluation completed.")
