from abc import abstractmethod

from ote.core.config import Config
from ote.logger import get_logger

logger = get_logger()

class IDataset:
    def __init__(self, data_config: Config):
        self._config = Config(data_config)
        self.datasets = None

    def get_subset(self, subset):
        logger.info(f"get_subset = {subset}")
        if self.datasets is None:
            logger.info("dataset was not built yet. builing...")
            self.build()
        dataset = self.datasets.get(subset)
        if dataset is None:
            logger.error(f"dataset doesn't have subset {subset}")
        return dataset

    @abstractmethod
    def build(self):
        raise NotImplementedError()

    @abstractmethod
    def update_config(self, config: dict):
        raise NotImplementedError()

    @property
    def config(self):
        return self._config