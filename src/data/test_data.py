from torchvision import datasets, transforms

class OCTTestData:
    def __init__(self, config):
        self.config = config
        self.dataset = datasets.ImageFolder(root=self.config.test_path)

        print(f"\nDataset loaded with {len(self.dataset)} images.")
        print(f"Found classes: {self.dataset.classes}")
        print(f"Class to index mapping: {self.dataset.class_to_idx}")
