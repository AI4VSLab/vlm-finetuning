import json
import torch
import numpy as np
from PIL import Image as PILImage
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms
from tfrecord.torch.dataset import TFRecordDataset


IMAGE_HEIGHT = 703
IMAGE_WIDTH = 1055
IMAGE_CHANNELS = 3


class OCTTranscriptDataset(Dataset):
    def __init__(self, path):
        # Initialize the tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

        self.augmentations = transforms.Compose(
            [
                # 1. Flip the image horizontally. This is the most effective augmentation.
                transforms.RandomHorizontalFlip(p=0.5),
                # 2. Apply a set of random geometric transformations.
                transforms.RandomAffine(
                    degrees=5,  # Rotate by +/- 10 degrees
                    translate=(
                        0.05,
                        0.05,
                    ),  # Shift horizontally and vertically by up to 5%
                    scale=(0.95, 1.05),  # Zoom in/out by up to 5%
                ),
                # 3. Adjust brightness and contrast to simulate different scanner settings.
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

        self.class_descriptions = {
            "AMD": "Wet Age-related Macular Degeneration (AMD)",
            "NORMAL": "Healthy eyes with no abnormalities (NORMAL)",
        }

        # Define the TFRecord schema description. This must match how the TFRecord
        # files were generated.
        tfrecord_description = {
            "input_ids": "int",  # Tokenized text input IDs
            "attn_mask": "int",  # Attention mask for text
            "class": "byte",  # Class label (e.g., 'n' for normal, 'a' for AMD)
            "normalized_image": "byte",  # Raw image bytes (float32, flattened)
        }

        # Initialize datasets for training and validation
        tfrecord_dataset = TFRecordDataset(
            path,
            None,
            tfrecord_description,
            transform=self._parse_tfrecord,
        )

        self.samples = [sample for sample in tfrecord_dataset]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        new_sample = {}

        new_sample["image"] = self.augmentations(sample["image"])
        new_sample["label"] = sample["label"]

        selected_version = torch.randint(0, 8, (1,)).item()
        input_ids = sample["input_ids"][selected_version]
        attn_mask = sample["attn_mask"][selected_version]
        biomarkers = self.tokenizer.decode(
            input_ids, attention_mask=attn_mask, skip_special_tokens=True
        )
        label = sample["label"]
        diagnosis_description = self.class_descriptions.get(label, label)
        target = {
            "diagnosis": diagnosis_description,
            "biomarkers": [feature.strip() for feature in biomarkers.split(";")],
        }
        new_sample["text"] = json.dumps(target)

        return new_sample

    def _parse_tfrecord(self, sample):
        new_sample = {}

        # 1. Image Processing: Convert byte array to PIL Image
        image_bytes = sample["normalized_image"]
        # Convert the byte array back to a numpy array of floats
        image = np.frombuffer(image_bytes, dtype=np.float32)

        # Reshape the flattened array to the correct image dimensions (Height, Width, Channels)
        image = image.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

        # Scale the float image data to the [0, 255] range and convert to uint8
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val > 1e-5:  # Avoid division by zero
            image = 255.0 * (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)  # If all pixels are same, create a black image

        image = image.astype(np.uint8)
        image = PILImage.fromarray(image)
        new_sample["image"] = image

        # 2. Transcript Tokens
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).reshape(8, -1)
        attn_mask = torch.tensor(sample["attn_mask"], dtype=torch.long).reshape(8, -1)
        new_sample["input_ids"] = input_ids
        new_sample["attn_mask"] = attn_mask

        # 3. Label Processing: Convert byte string label to integer (0 or 1)
        class_str = sample["class"].decode("utf-8")
        label = (
            "NORMAL" if class_str == "n" else "AMD"
        )  # 'n' for normal (0), 'a' for AMD (1)
        new_sample["label"] = label

        return new_sample
