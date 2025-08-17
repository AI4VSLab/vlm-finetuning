import logging
from torchvision import datasets

logger = logging.getLogger(__name__)


class OCTTestDataset(datasets.ImageFolder):
    def __init__(self, path):
        datasets.ImageFolder.__init__(self, root=path)

        logger.info(f"Test Dataset loaded with {len(self)} images.")
        logger.info(f"Found classes: {self.classes}")
        logger.info(f"Class to index mapping: {self.class_to_idx}")
