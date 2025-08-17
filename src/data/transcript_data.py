import logging
from datasets import OCTTranscriptDataset, OCTTestDataset

logger = logging.getLogger(__name__)


class OCTTranscriptData:
    def __init__(self, config):
        logger.info("Loading Transcript Data... ")
        self.train_data = OCTTranscriptDataset(config.train_path, config.prompt)
        self.test_data = OCTTestDataset(config.test_path)
        logger.info("Data Loaded.")
