from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger as PLLogger
import lightning.pytorch as pl


class BaseLogger(PLLogger):

    def after_remove_checkpoint(self, filepath: str):
        pass
