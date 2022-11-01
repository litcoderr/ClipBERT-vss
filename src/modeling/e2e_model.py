from src.modeling.modeling import (
    ClipBertForPreTraining,
    ClipBertForSequenceClassification,
    ClipBertForMultipleChoice,
    ClipBertForRegression,
    OursForVideoTextRetrieval,
    ClipBertForVideoTextRetrieval)
from src.modeling.grid_feat import GridFeatBackbone
from torch import nn
from src.datasets.data_utils import repeat_tensor_rows
from src.utils.load_save import load_state_dict_with_mismatch


class Ours(nn.Module):
    def __init__(self, config, transformer_cls=OursForVideoTextRetrieval):
        super(Ours, self).__init__()
        self.config = config
        self.transformer = transformer_cls(config)

    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        del batch["n_examples_list"]

        visual_features = batch["visual_inputs"]
        batch["visual_inputs"] = repeat_tensor_rows(
            visual_features, repeat_counts)

        batch["sample_size"] = len(repeat_counts)  # batch size

        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, bert_weights_path=None):
        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)


class ClipBert(nn.Module):
    def __init__(self, config, input_format="BGR",
                 detectron2_model_cfg=None,
                 transformer_cls=ClipBertForPreTraining):
        super(ClipBert, self).__init__()
        self.config = config
        self.detectron2_model_cfg = detectron2_model_cfg
        assert detectron2_model_cfg is not None
        cnn_cls = GridFeatBackbone
        print(f"cnn_cls {cnn_cls}")
        self.cnn = cnn_cls(
            detectron2_model_cfg=detectron2_model_cfg,
            config=config, input_format=input_format)
        self.transformer = transformer_cls(config)
        self.retrieval = transformer_cls == ClipBertForVideoTextRetrieval

    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        del batch["n_examples_list"]
        visual_features = self.cnn(batch["visual_inputs"])
        batch["visual_inputs"] = repeat_tensor_rows(
            visual_features, repeat_counts)
        if self.retrieval:
            batch["sample_size"] = len(repeat_counts)  # batch size
        # batch["visual_inputs"] shape: [n_examples*batches, n_frames, 7, 7, hidden_dim(768)]
        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False
