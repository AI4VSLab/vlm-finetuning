import logging
from torch.utils.data import ConcatDataset
from .datasets import OCTTranscriptDataset, OCT5KDataset, OCTAMDDataset

logger = logging.getLogger(__name__)


class OCTTranscriptData:
    def __init__(self, config):
        logger.info("Loading Biomarker Data... ")
        transcript_dataset = OCTTranscriptDataset(config.oct_transcript_path)
        oct5k_dataset = OCT5KDataset(config.oct_5k_images_path, config.oct_5k_csv_path)
        self.train_data = ConcatDataset([transcript_dataset, oct5k_dataset])
        logger.info(
            f"Biomarker dataset loaded with {len(self.train_data)} images."
        )

        self.test_data = OCTAMDDataset(root=config.test_path)
        logger.info(f"AMD Test Dataset loaded with {len(self.test_data)} images.")
        logger.info(f"Found classes: {self.test_data.classes}")
        logger.info(f"Class to index mapping: {self.test_data.class_to_idx}")
        logger.info("Transcript Data Loaded.")
