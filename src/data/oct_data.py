import logging
import torch
import numpy as np
from PIL import Image as PILImage
from transformers import AutoTokenizer
from tfrecord.torch.dataset import TFRecordDataset

logger = logging.getLogger(__name__)

IMAGE_HEIGHT = 703
IMAGE_WIDTH = 1055
IMAGE_CHANNELS = 3

class OCTExpertData:
    def __init__(self, config):
        logger.info("Loading OCTExpert data...")
        self.config = config

        # Initialize the tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

        # Define the TFRecord schema description. This must match how the TFRecord
        # files were generated.
        tfrecord_description = {
            "input_ids": "int",        # Tokenized text input IDs
            "attn_mask": "int",        # Attention mask for text
            "class": "byte",           # Class label (e.g., 'n' for normal, 'a' for AMD)
            "normalized_image": "byte" # Raw image bytes (float32, flattened)
        }

        logger.info(f"Train TFRecord path: {self.config.train_path}")

        # Initialize datasets for training and validation
        self.tfrecord_dataset = TFRecordDataset(
            self.config.train_path,
            None,
            tfrecord_description,
            transform=self._preprocess_sample
        )

        self.prompt = """You are a highly skilled AI assistant specializing in ophthalmic diagnostics. Your task is to analyze Optical Coherence Tomography (OCT) images to determine the presence of Age-related Macular Degeneration (AMD) or if the image appears normal. Provide a concise explanation of your findings and a definitive diagnosis.
        Your output must strictly follow this format:
        <explanation>[Your detailed but concise reasoning based on OCT findings]</explanation> <diagnosis>[AMD or Normal]</diagnosis>
        """
        # Create the processed dataset with conversation format
        self.processed_dataset = [self._convert_to_conversation(sample) for sample in self.tfrecord_dataset]

        # For compatibility with Trainer/Evaluator, self.dataset points to the conversation-formatted data
        self.dataset = self.processed_dataset
        logger.info("OCTExpert data initialized.")

    def _preprocess_sample(self, sample):
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
        if max_val - min_val > 1e-5: # Avoid division by zero
            image = 255.0 * (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image) # If all pixels are same, create a black image

        image = image.astype(np.uint8)
        new_sample["image"] = PILImage.fromarray(image)

        # 2. Text Decoding: Convert input_ids and attention_mask back to text
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(sample["attn_mask"], dtype=torch.long)
        decoded_text = self.tokenizer.decode(input_ids, attention_mask=attn_mask, skip_special_tokens=True)
        new_sample["text"] = decoded_text

        # 3. Label Processing: Convert byte string label to integer (0 or 1)
        class_str = sample["class"].decode('utf-8')
        label = 0 if class_str == 'n' else 1 # 'n' for normal (0), 'a' for AMD (1)
        new_sample["label"] = label

        return new_sample

    def _convert_to_conversation(self, sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ("<explanation>" + sample["text"]+"</explanation> <diagnosis>" + ("AMD" if sample['label'] else "Normal") +"</diagnosis>")}]},
        ]
        return {"messages": conversation}
