from typing import Dict, Any
import os
import json
import torch
import random
import lightning as pl
from transformers import BertConfig, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Pad, ToTensor
from enum import Enum
from pathlib import Path
from PIL import Image

from src.modeling.e2e_model import ClipBertforStar
from src.modeling.modeling import ClipBertBaseModel
from src.configs.config import shared_configs
from src.utils.basic_utils import load_json
from src.utils.load_save import load_state_dict_with_mismatch


class DatasetType(Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2


DATASET_PATH_ROOT = Path("/txt")
SPLIT_PATH_ROOT = DATASET_PATH_ROOT / "Questions, Answers and Situation Graphs"
VID_PATH_ROOT= DATASET_PATH_ROOT / "Situation Video Data" / "Keyframe Dumping Tool from Action Genome" / "dataset" / "ag" / "frames"
class StarDataset(Dataset):
    def __init__(self, config, type: DatasetType):
        self.config = config
        self.type = type
        self.split_info = self.get_split_info()

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, index):
        video_id = self.split_info[index]["video_id"]
        question = self.split_info[index]["question"]
        choices = [c["choice"] for c in self.split_info[index]["choices"]]
        answer_index = choices.index(self.split_info[index]["answer"])

        # sample video frame (random)
        video_path = VID_PATH_ROOT / f"{video_id}.mp4"
        frame_list = os.listdir(video_path)
        frame_list = random.sample(frame_list, self.config.train_n_clips)
        frame_list.sort()

        # load video frame
        frames = [Image.open(video_path / name) for name in frame_list]

        # resize and pad
        for idx, frame in enumerate(frames):
            # resize so that the larger dim is equal to max_img_size
            frame = Resize(self.config.max_img_size-1, max_size=self.config.max_img_size)(frame)

            # pad shorter dim is eqal to max_img_size
            w, h = frame.size
            padding = max(w, h) - min(w, h)
            if padding % 2 == 0:
                p1 = padding // 2
                p2 = padding // 2
            else:
                p1 = padding // 2
                p2 = padding // 2 + 1
            if w < h:
                frame = Pad((p1, 0, p2, 0), fill=0, padding_mode='constant')(frame)
            elif w > h:
                frame = Pad((0, p1, 0, p2), fill=0, padding_mode='constant')(frame)
            
            frames[idx] = ToTensor()(frame)
        # [frame_len(8), c(3), h(448), w(448)]
        frames = torch.stack(frames, dim=0)

        item = {
            "video": frames,
            "question": question,
            "choices": choices,
            "answer_index": answer_index
        }
        return item

    def get_split_info(self) -> Dict[str, Any]:
        if self.type == DatasetType.TRAIN:
            json_path = SPLIT_PATH_ROOT / "STAR_train.json"
        elif self.type == DatasetType.VAL:
            json_path = SPLIT_PATH_ROOT / "STAR_val.json"
        else:
            json_path = SPLIT_PATH_ROOT / "STAR_test.json"
        
        with open(json_path, "r") as f:
            split_info = json.load(f)
        
        return split_info

class StarDatasetCollator():
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.tokenizer_dir)
        self.ques_max_length = self.config.max_txt_len
        self.choice_max_length = 10
    
    def collate_fn(self, batch):
        b_size = len(batch)

        #### tokenize question #####
        questions = [b["question"] for b in batch]
        tokenized = self.tokenizer.batch_encode_plus(
            questions,
            max_length=self.ques_max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        # [b*4, max_len(20)]
        ques_tokenid = tokenized.input_ids.unsqueeze(1).repeat(1, 4, 1).reshape(4*b_size, self.ques_max_length)  
        # [b*4, max_len(20)]
        ques_mask = tokenized.attention_mask.unsqueeze(1).repeat(1,4,1).reshape(4*b_size, self.ques_max_length)

        #### tokenize choices ####
        choices = []
        for b in batch:
            choices += b["choices"]
        tokenized = self.tokenizer.batch_encode_plus(
            choices,
            max_length=self.choice_max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        # [b*4, max_len(10)]
        choice_tokenid = tokenized.input_ids
        # [b*4, max_len(10)]
        choice_mask = tokenized.attention_mask

        #### build token ####
        # [b*4, max_len(30)]
        token_id = torch.cat([ques_tokenid, choice_tokenid], dim=1)
        # [b*4, max_len(30)]
        token_mask = torch.cat([ques_mask, choice_mask], dim=1)

        #### generate frame batch ####
        # [b, frame_len(8), c, h, w]
        frame = torch.stack([b["video"] for b in batch], dim=0)
        _, f_len, c, h, w = tuple(frame.shape)
        # [b*4, frame_len(8), c, h, w]
        frame = frame.unsqueeze(1).repeat(1, 4, 1, 1, 1, 1).reshape(4*b_size, f_len, c, h, w)

        return {
            "token_id": token_id,
            "token_mask": token_mask,
            "frame": frame
        }


class StarQAClipbert(pl.LightningModule):
    def __init__(self, config):
        super(StarQAClipbert, self).__init__()
        self.config = config

        # initialize model
        model_config = BertConfig(**load_json(self.config.model_config))
        self.model = ClipBertforStar(
            config=model_config,
            input_format=self.config.img_input_format,
            detectron2_model_cfg=self.config.detectron2_model_cfg,
            transformer_cls=ClipBertBaseModel
        )
        load_state_dict_with_mismatch(self.model, self.config.e2e_weights_path)
        self.model.to(self.config.device)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        y_ = self.forward(train_batch)

    def validation_step(self, val_batch, batch_idx):
        pass


if __name__ == "__main__":
    # initialize star clipbert pl module
    config = shared_configs.get_starqa_args()
    model = StarQAClipbert(config)

    # initialize dataloader
    collator = StarDatasetCollator(config=config)
    train_dataset = StarDataset(config, type=DatasetType.TRAIN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=collator.collate_fn
    )
    val_dataset = StarDataset(config, type=DatasetType.VAL)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=True,
        collate_fn=collator.collate_fn
    )

    # initialize pl trainer
    trainer = pl.Trainer(gpus=1)

    # train
    trainer.fit(model, train_loader, val_loader)
