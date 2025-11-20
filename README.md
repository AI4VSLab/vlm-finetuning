# OCT Biomarker VLM Training

This repository implements the Vision-Language Model (VLM) component of the paper **"Toward an AI Co-Annotator for Interactive Clinical Diagnosis"**.

It fine-tunes **MedGemma** (a medical-specific VLM) using **Unsloth** and **LoRA** to act as an expert ophthalmology assistant. The model is designed to provide "ontology-bounded" biomarker summaries i.e. it detects specific, clinically relevant features rather than generating free-text descriptions.

## Key Features

  * **Model Architecture**: Finetuned `unsloth/medgemma-4b-it` using 4-bit Quantized LoRA (Low-Rank Adaptation) for efficient training on consumer GPUs.
  * **Two-Stage Curriculum Learning**:
    1.  **Foundation Stage**: Trains on large-scale binary diagnosis (Normal vs. Wet AMD) and VQA tasks to learn visual features.
    2.  **Unified Stage**: Adds the complex "Biomarker List Generation" task while maintaining performance on diagnosis.
  * **Balanced Sampling**: Implements a custom `WeightedRandomSampler` to handle severe class imbalances between "Normal" and "Pathological" samples across different tasks.
  * **Strict Biomarker Ontology**: The model is trained to recognize a specific schema of 18 retinal biomarkers (e.g., *Drusen, Intraretinal Fluid, Choroidal Neovascularization*) to ensure clinical precision.

## Tasks

The model learns to solve three distinct tasks simultaneously:

1.  **Diagnosis**: `[DIAGNOSIS] What is the diagnosis...?` $\rightarrow$ `Wet Age-related Macular Degeneration`
2.  **VQA (Visual Question Answering)**: `[VQA] Is Subretinal Fluid present...?` $\rightarrow$ `Yes`
3.  **List Generation**: `[BIOMARKERS] What biomarkers are visible...?` $\rightarrow$ `Drusen, Subretinal Fluid, Hemorrhage`

## Installation

This project uses `uv` for fast dependency management.

```bash
make install
```

## Data Structure

The project relies on Hydra for configuration. Ensure your data is organized as follows and update `src/config/train.yaml` with your paths:

  * **Diagnosis Data**: Folder structure compatible with `torchvision.datasets.ImageFolder` (e.g., `root/train/CNV/x.jpg`).
  * **Biomarker Data**: A folder of images and a CSV file containing columns for `image_filename` and `biomarkers` (list of strings).

## Usage

### 1\. Training

The training pipeline uses `hydra` to manage configurations. You can use the Makefile shortcuts:

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

Evaluates the model on held-out test sets using **Accuracy** (for Diagnosis/VQA) and **BERTScore/MedBERTScore** (for Biomarker text generation).

```bash
make test
```

## Project Layout

  * `src/algorithm/`: Custom `BalancedSFTTrainer` and `Evaluator` logic.
  * `src/data/`: Dataset classes (`OCTDiagnosis`, `OCTBiomarkers`, `OCTVQA`) and sampling logic.
  * `src/model/`: MedGemma wrapper using `Unsloth` and `Peft`.
  * `src/config/`: Hyperparameters for training (`train.yaml`) and evaluation (`evaluate.yaml`).

## Citation

> **Toward an AI Co-Annotator for Interactive Clinical Diagnosis: Gaze-Aligned ViT AOI Guidance and Ontology-Bounded VLM Biomarker Drafting in Age-Related Macular Degeneration**.
