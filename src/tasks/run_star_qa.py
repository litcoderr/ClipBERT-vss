import torch
import lightning as pl
from transformers import BertConfig

from src.modeling.e2e_model import ClipBert
from src.modeling.modeling import ClipBertBaseModel
from src.configs.config import shared_configs
from src.utils.basic_utils import load_json


class StarQAClipbert(pl.LightningModule):
    def __init__(self, config):
        self.config = config

        # initialize model
        model_config = BertConfig(**load_json(self.config.model_config))
        model = ClipBert(
            config=model_config,
            input_format=self.config.img_input_format,
            detectron2_model_cfg=self.config.detectron2_model_cfg,
            transformer_cls=ClipBertBaseModel
        )
        pass

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass


if __name__ == "__main__":
    # initialize star clipbert pl module
    config = shared_configs.get_starqa_args()
    model = StarQAClipbert(config)

    # initialize dataloader

    # initialize pl trainer
    trainer = pl.Trainer(gpus=1)

    # trainer.fit(model, train_loader, val_loader)
