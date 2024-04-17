"""
A pipeline that uses only one view/image. It gets predictions from a model (supports multi-outputs ex points AND lines).
It gets the predictions and compares it directly with the ground-truth for loss (gt generated by teachers) -> Supervised
"""
from omegaconf import OmegaConf

from . import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class OneViewPipeline(BaseModel):
    def _init(self, conf):
        pass

    def _forward(self, data):
        pass

    def loss(self, pred, data):
        pass


