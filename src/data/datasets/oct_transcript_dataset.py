import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms
from tfrecord.torch.dataset import TFRecordDataset

logger = logging.getLogger(__name__)

IMAGE_HEIGHT = 703
IMAGE_WIDTH = 1055
IMAGE_CHANNELS = 3


class OCTTranscriptDataset(Dataset):
    def __init__(self, path, prompt):
        self.prompt = prompt

        # Initialize the tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

        self.augmentations = transforms.Compose(
            [
                # 1. Flip the image horizontally. This is the most effective augmentation.
                transforms.RandomHorizontalFlip(p=0.5),
                # 2. Apply a set of random geometric transformations.
                transforms.RandomAffine(
                    degrees=10,  # Rotate by +/- 10 degrees
                    translate=(
                        0.05,
                        0.05,
                    ),  # Shift horizontally and vertically by up to 5%
                    scale=(0.95, 1.05),  # Zoom in/out by up to 5%
                ),
                # 3. Adjust brightness and contrast to simulate different scanner settings.
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # 4. Convert PIL Image to a tensor.
                transforms.ToTensor(),
                # 5. Normalize the tensor.
                #    These are standard ImageNet stats. You might need to adjust them
                #    based on your model's requirements or dataset statistics.
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Define the TFRecord schema description. This must match how the TFRecord
        # files were generated.
        tfrecord_description = {
            "input_ids": "int",  # Tokenized text input IDs
            "attn_mask": "int",  # Attention mask for text
            "class": "byte",  # Class label (e.g., 'n' for normal, 'a' for AMD)
            "normalized_image": "byte",  # Raw image bytes (float32, flattened)
        }

        logger.info(f"Train TFRecord path: {path}")

        # Initialize datasets for training and validation
        tfrecord_dataset = TFRecordDataset(
            path,
            None,
            tfrecord_description,
            transform=self._parse_tfrecord,
        )

        self.samples = [sample for sample in tfrecord_dataset]

        logger.info(f"Transcript dataset loaded with {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        new_sample = {}

        new_sample["image"] = self.augmentations(sample["image"])

        selected_version = torch.randint(0, 8, (1,)).item()
        input_ids = sample["input_ids"][selected_version]
        attn_mask = sample["attn_mask"][selected_version]
        decoded_text = self.tokenizer.decode(
            input_ids, attention_mask=attn_mask, skip_special_tokens=True
        )
        new_sample["text"] = decoded_text

        # 3. Convert to conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image", "image": new_sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": new_sample["text"]}],
            },
        ]
        new_sample["messages"] = conversation

        return new_sample
