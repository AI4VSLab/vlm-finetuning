import json
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os


class OCTC8Dataset(Dataset):
    def __init__(self, root, split="train"):
        """
        Args:
            root (string): Root directory path of the dataset.
            split (string): Specifies the dataset split ('train', 'val', 'test').
        """
        self.prompt = "Analyze this retinal OCT scan and provide a diagnosis and a list of key biomarkers in JSON format."
        self.split = split
        self.dataset = datasets.ImageFolder(os.path.join(root, split))
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

        # Dictionary to map class names to descriptions
        self.class_descriptions = {
            "AMD": "Age-related Macular Degeneration",
            "CNV": "Choroidal Neovascularization",
            "CSR": "Central Serous Retinopathy",
            "DME": "Diabetic Macular Edema",
            "DR": "Diabetic Retinopathy",
            "DRUSEN": "Yellow deposits under the retina",
            "MH": "Macular Hole",
            "NORMAL": "Healthy eyes with no abnormalities",
        }

        # Define augmentations for the training split
        if self.split == "train":
            self.augmentations = transforms.Compose(
                [
                    # 1. Flip the image horizontally.
                    transforms.RandomHorizontalFlip(p=0.5),
                    # 2. Apply random geometric transformations.
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                    ),
                    # 3. Adjust brightness and contrast.
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ]
            )
        else:
            # For validation or test, just return the image without augmentation
            self.augmentations = transforms.Compose([])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Get the raw image and label from the internal ImageFolder dataset
        image, label_idx = self.dataset[idx]
        label_name = self.classes[label_idx]

        # 2. Apply augmentations
        image = self.augmentations(image)

        # 3. Create the structured text output
        # Since this dataset only has labels, we'll create a JSON structure
        # with an empty list for biomarkers to maintain consistency.
        # Format the diagnosis as "<description> (<class>)"
        diagnosis_description = self.class_descriptions.get(
            label_name
        )  # Get description, default to class name if not found
        target = {
            "diagnosis": f"{diagnosis_description} ({label_name})",
            "biomarkers": [],  # No biomarkers in this dataset
        }
        text_output = json.dumps(target)

        # 4. Format the data into the final dictionary
        sample = {
            "image": image,
            "label": label_name,
            "text": text_output,
        }

        # 5. Convert to the conversational format for the model
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
        sample["messages"] = conversation

        return sample
