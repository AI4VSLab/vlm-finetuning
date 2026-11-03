# OCT Biomarker VLM Training

This repository implements the Vision-Language Model (VLM) component of the paper **"Toward an AI Co-Annotator for Interactive Clinical Diagnosis"**.

It fine-tunes **MedGemma** using **Unsloth** to serve as an expert ophthalmology assistant, capable of diagnosing AMD and identifying specific retinal biomarkers.

## Data

The project utilizes three distinct dataset types to train the model:

  * **Diagnosis**: Binary classification data (Normal vs. Wet AMD).
  * **VQA**: Pairs of images and specific "Yes/No" questions about biomarker presence.
  * **Biomarker List**: Images paired with a comma-separated list of all visible pathological findings.

To handle the severe class imbalance often found in medical data (e.g., far more "No" answers than "Yes"), the data loader implements a custom `WeightedRandomSampler` to ensure balanced exposure to all classes during training.

## Model

The core model is **MedGemma** (a medical VLM) optimized via **Unsloth** for efficient training.

  * **Architecture**: The model is fine-tuned using **4-bit Quantized LoRA** (Low-Rank Adaptation), allowing it to run on consumer-grade GPUs while retaining high accuracy.
  * **Tasks**: The model is trained to perform three tasks simultaneously:
    1.  **Diagnosis**: Predicting "Wet Age-related Macular Degeneration" or "Normal".
    2.  **VQA**: Answering targeted questions like "Is Subretinal Fluid present?".
    3.  **List Generation**: Outputting a structured list of findings (e.g., "Drusen, Hemorrhage").
  * **Ontology-Bounded**: The model is constrained to recognize a specific schema of 18 clinically relevant biomarkers, preventing the generation of vague or hallucinated text.

## Algorithm

The training strategy employs a **Two-Stage Curriculum** to ensure optimal performance:

1.  **Foundation Stage**: The model is first trained on large-scale binary Diagnosis and VQA tasks to build a strong visual understanding of retinal pathology.
2.  **Unified Stage**: The complex "Biomarker List Generation" task is introduced. The model is trained on all three tasks together to refine its ability to detect and list multiple findings.

Evaluation is performed using a custom `Evaluator` that calculates Accuracy for diagnosis and semantic metrics (BLEU, METEOR, BERTScore) for the generated biomarker lists.

## Installation

This project uses `uv` for fast dependency management.

```bash
make install
```

## Usage

### 1\. Training

The training pipeline uses `hydra` for configuration. You can use the Makefile shortcuts:

```bash
# Train the model (Default configuration)
make train
```

To train specific stages manually (recommended for the curriculum approach):

```bash
# Step 1: Train Foundation (Diagnosis + VQA)
python src/train.py data.stage=foundation

# Step 2: Train Unified (All Tasks)
python src/train.py data.stage=unified
```

### 2\. Evaluation

Evaluates the model on held-out test sets using Accuracy and text-generation metrics.

```bash
make test
```

## Citation


## Citation


## Citation

If you use this repository, the associated dataset, or the trained model in your research, please cite our paper:

**Co-Annotator: Expert-Distilled ViT and VLM for Visual and Documentation Guidance in Age-Related Macular Degeneration**

*ACM Symposium on User Interface Software and Technology (UIST '26)*

```bibtex
@inproceedings{li2026coannotator,
  title     = {Co-Annotator: Expert-Distilled ViT and VLM for Visual and Documentation Guidance in Age-Related Macular Degeneration},
  author    = {Li, Ziheng and Freeman, Benjamin and Raman, Akshay and Rajkumar, Kavin Aravindhan and Fang, Xinxin and Srivastava, Rishabh and Feiner, Steven and Thakoor, Kaveri A.},
  booktitle = {Proceedings of the 39th Annual ACM Symposium on User Interface Software and Technology (UIST '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3830398.3830722},
  url       = {https://doi.org/10.1145/3830398.3830722}
}
```

**Paper:** [arXiv](https://arxiv.org/abs/2608.30352) | [ACM DOI](https://doi.org/10.1145/3830398.3830722)
