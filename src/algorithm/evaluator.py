import logging
import random
from tqdm import tqdm
import numpy as np
import evaluate
import nltk

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, data, config):
        self.config = config
        self.model = model
        self.data = data

        logger.info("Loading text generation metrics (BLEU, METEOR, ROUGE)...")
        try:
            nltk.download("punkt", quiet=True)
        except Exception as e:
            logger.warning(
                f"NLTK download for 'punkt' failed: {e}. METEOR score might not work."
            )

        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        self.rouge = evaluate.load("rouge")
        self.bert = evaluate.load("bertscore")

        logger.info("Evaluator initialized for evaluation and sample generation.")

    def _calculate_lexical_metrics(self, predictions, references):
        bleu_score = self.bleu.compute(predictions=predictions, references=references)
        meteor_score = self.meteor.compute(
            predictions=predictions, references=references
        )
        rouge_scores = self.rouge.compute(
            predictions=predictions, references=references
        )
        return bleu_score, meteor_score, rouge_scores

    def _calculate_semantic_metrics(self, predictions, references):
        bert_scores = self.bert.compute(
            predictions=predictions, references=references, lang="en"
        )
        medbert_scores = self.bert.compute(
            predictions=predictions,
            references=references,
            model_type="emilyalsentzer/Bio_ClinicalBERT",
            num_layers=12,
        )
        return bert_scores, medbert_scores

    def _calculate_accuracy(self, predictions, references):
        correct = 0
        total = len(predictions)
        for pred, ref in zip(predictions, references):
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
        return {"accuracy": correct / total if total > 0 else 0}

    def evaluate(self, dataset, num_samples=50, task_type="diagnosis"):
        all_samples = dataset
        sample_indices = random.sample(
            range(len(all_samples)), min(num_samples, len(all_samples))
        )

        all_predicted_texts = []
        all_reference_texts = []

        for idx in tqdm(sample_indices, desc=f"Evaluating {task_type}"):
            sample = all_samples[idx]
            image = sample["image"]
            prompt = sample["instruction"]
            reference = sample["response"]

            predicted = self.model.generate(prompt, image, use_streamer=False)

            all_predicted_texts.append(predicted)
            all_reference_texts.append(reference)

        if task_type == "biomarkers":
            lexical_metrics = self._calculate_lexical_metrics(
                all_predicted_texts, all_reference_texts
            )
            semantic_metrics = self._calculate_semantic_metrics(
                all_predicted_texts, all_reference_texts
            )
            return lexical_metrics, semantic_metrics
        else:
            return self._calculate_accuracy(all_predicted_texts, all_reference_texts)

    def generate(self, dataset, num_samples=5):
        all_samples = dataset
        sample_indices = random.sample(
            range(len(all_samples)), min(num_samples, len(all_samples))
        )

        for i, idx in enumerate(sample_indices):
            try:
                sample = all_samples[idx]
                image = sample["image"]
                prompt = sample["instruction"]
                reference = sample["response"]

                if image is None:
                    logger.warning(f"Skipping sample {idx} due to missing image.")
                    continue

                logger.info(
                    f"\n--- Sample {i + 1}/{num_samples} (Original Index: {idx}) ---"
                )
                logger.info(f"Instruction: {prompt}")
                logger.info(f"Reference Response: {reference}")
                logger.info("Predicted Response:")

                print(self.model.generate(prompt, image, use_streamer=True))

            except Exception as e:
                logger.error(f"Error generating prediction for index {idx}: {e}")
                continue

    def run(self):
        logger.info("Starting evaluation and sample generation process...")

        eval_samples = self.config.eval_samples
        gen_samples = self.config.gen_samples

        # Diagnosis Evaluation
        logger.info("\n--- Diagnosis Task ---")
        diag_accuracy = self.evaluate(
            self.data.diagnosis_dataset, num_samples=eval_samples, task_type="diagnosis"
        )
        logger.info(f"Accuracy: {diag_accuracy['accuracy']:.4f}")
        self.generate(self.data.diagnosis_dataset, num_samples=gen_samples)

        # Biomarkers Evaluation
        logger.info("\n--- Biomarkers Task ---")
        (bio_bleu, bio_meteor, bio_rouge), (bio_bert, bio_medbert) = self.evaluate(
            self.data.biomarkers_dataset,
            num_samples=eval_samples,
            task_type="biomarkers",
        )
        logger.info(
            f"BLEU: {bio_bleu['bleu']:.4f}, METEOR: {bio_meteor['meteor']:.4f}, ROUGE-L: {bio_rouge['rougeL']:.4f}"
        )
        logger.info(
            f"BERT Score: {np.mean(bio_bert['f1']):.4f}, MedBERT Score: {np.mean(bio_medbert['f1']):.4f}"
        )
        self.generate(self.data.biomarkers_dataset, num_samples=gen_samples)

        # VQA Evaluation
        logger.info("\n--- VQA Task ---")
        vqa_accuracy = self.evaluate(
            self.data.vqa_dataset, num_samples=200, task_type="vqa"
        )
        logger.info(f"Accuracy: {vqa_accuracy['accuracy']:.4f}")
        self.generate(self.data.vqa_dataset, num_samples=gen_samples)

        # Test Dataset Evaluation
        logger.info("\n--- Test Task ---")
        test_accuracy = self.evaluate(
            self.data.test_dataset, num_samples=100, task_type="diagnosis"
        )
        logger.info(f"Accuracy: {test_accuracy['accuracy']:.4f}")
        self.generate(self.data.test_dataset, num_samples=gen_samples)

        logger.info("\nEvaluation completed.")
