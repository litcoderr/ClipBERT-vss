import random
import copy
import os
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from src.datasets.dataset_base import ClipBertBaseDataset


class OursVideoRetrievalDataset(ClipBertBaseDataset):
    """ This should work for both train and test (where labels are not available).
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    itm_neg_prob: float [0, 1] set to 0 will disable itm.
    random_sample_clips: bool, whether using randomly sampled N clips or always use uniformly sampled N clips
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir, vss_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=40, itm_neg_size=1,
                 ensemble_n_clips=1, random_sample_clips=True):
        super(OursVideoRetrievalDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.vss_dir = vss_dir
        self.ensemble_n_clips = ensemble_n_clips
        self.num_labels = 2
        self.itm_neg_size = itm_neg_size
        self.random_sample_clips = random_sample_clips
        self.id2data = {
            d["id"]: d for group in datalist for d in group[1]}

    def __len__(self):
        return len(self.datalist)

    def _get_npy_list(self, vid_id):
        path = os.path.join(self.vss_dir, vid_id)
        npy_list = [os.path.join(path, npy_name) for npy_name in os.listdir(path)]
        npy_list.sort()
        return npy_list

    def _load_npy_random(self, vid_id):
        npy_list = self._get_npy_list(vid_id)

        start_idx = random.randrange(0, len(npy_list)-self.num_frm)
        idx_list = [start_idx + i for i in range(self.num_frm)]
        sampled_npy_list = [npy_list[idx] for idx in idx_list] 

        result = []
        for npy_path in sampled_npy_list:
            npy = np.load(npy_path)
            result.append(torch.tensor(npy).unsqueeze(0))
        return torch.cat(result, dim=0)

    def _load_video_multi_clips_random(self, vid_id):
        """take multiple clips at fixed position"""
        vid_frm_array_list = []
        for clip_idx in range(self.ensemble_n_clips):
            frames = self._load_npy_random(vid_id)
            vid_frm_array_list.append(frames)
        return torch.cat(vid_frm_array_list, dim=0)

    def _load_video_multi_clips_uniform(self, vid_id):
        """
        vid_frm_array_list = []
        video_max_pts = None
        for clip_idx in range(self.ensemble_n_clips):
            frames, video_max_pts = self._load_video(
                vid_id,
                num_clips=self.ensemble_n_clips,
                clip_idx=clip_idx, safeguard_duration=True,
                video_max_pts=video_max_pts)
            vid_frm_array_list.append(frames)
        return None if any([e is None for e in vid_frm_array_list]) else torch.cat(vid_frm_array_list, dim=0)
        """
        pass

    def __getitem__(self, index):
        # skip error videos:
        # read video representation
        vid_id, examples = self.datalist[index]  # one video with multiple examples
        if self.ensemble_n_clips > 1:
            if self.random_sample_clips:
                # [n_clips * num_frm, 768]
                vid_frm_array = self._load_video_multi_clips_random(vid_id)
                pass
            else:
                # vid_frm_array = self._load_video_multi_clips_uniform(vid_id)
                pass
        else:
            if self.random_sample_clips:
                # vid_frm_array, _ = self._load_video(vid_id)  # tensor (T, C, H, W)
                pass
            else:
                # vid_frm_array, _ = self._load_video(vid_id, num_clips=1, clip_idx=0)  # tensor (T, C, H, W)
                pass

        sampled_examples = []
        for e in examples:
            s = self._get_single_example(e, index)
            if isinstance(s, dict):
                sampled_examples.append(s)
            else:
                sampled_examples.extend(s)

        return dict(
            vid=vid_frm_array,
            examples=sampled_examples,
            n_examples=len(sampled_examples)  # used to create image feature copies.
        )

    def _get_single_example(self, data, index):
        examples = []

        text_str = data["txt"]
        itm_label = 1  # positive pair
        examples.append(dict(
            text_str=text_str,
            itm_label=itm_label
        ))
        count = 0
        while self.itm_neg_size > count:
            text_str = self._get_random_negative_caption(index)
            itm_label = 0  # negative pair
            examples.append(dict(
                text_str=text_str,
                itm_label=itm_label
            ))
            count += 1
        return examples

    def _get_random_negative_caption(self, gt_index):
        gt_img_id, _ = self.datalist[gt_index]
        neg_img_id = gt_img_id
        while neg_img_id == gt_img_id:
            neg_index = int(random.random() * len(self))
            neg_img_id, neg_examples = self.datalist[neg_index]
        neg_data = neg_examples[int(random.random() * len(neg_examples))]
        return neg_data["txt"]


class VideoRetrievalCollator(object):
    def __init__(self, tokenizer, max_length=40):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])  # (sum(n_examples_list)) ex [{}, {}, ... {}]
        n_examples_list = [d["n_examples"] for d in batch]  # (B, ) ex [2,2,2,2,2....,2]
        # group elements data
        # directly concatenate question and option as a single seq.
        text_str_list = [d["text_str"] for d in text_examples]  #[B * n_examples]
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (sum(n_examples_list), L)
        text_input_mask = batch_enc.attention_mask  # (sum(n_examples_list), L)

        if "itm_label" in text_examples[0]:
            itm_labels = default_collate(
                [d["itm_label"] for d in text_examples])  # (sum(n_examples_list), )
        else:
            itm_labels = None

        if "id" in text_examples[0]:
            caption_ids = [d["id"] for d in text_examples]  # (sum(n_examples_list), )
        else:
            caption_ids = None
        collated_batch = dict(
            visual_inputs=visual_inputs,  # (B, #frm, H, W, C)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            caption_ids=caption_ids,  # list(int), example ids,
            labels=itm_labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
        if "vid_id" in batch[0] and len(batch) == 1:
            collated_batch["vid_id"] = batch[0]["vid_id"]
        return collated_batch


class OursVideoRetrievalEvalDataset(ClipBertBaseDataset):
    """ Sample by video/image, calculate scores between each video with all the text
    and loop through all the videos. Each batch will only contain a single video,
    but multiple text.

    datalist: list(dict), each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir, vss_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=40, ensemble_n_clips=1):
        self.ensemble_n_clips = ensemble_n_clips
        super(OursVideoRetrievalEvalDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        # id is unique id per caption/example
        self.vss_dir = vss_dir

        for i, d in enumerate(self.datalist):
            assert i == d["id"]
        self.gt_cap_id2vid_id = {d["id"]: d["vid_id"] for d in datalist}
        self.cap_id2data = {d["id"]: d for d in datalist}
        self.batches = self._prepare_batches_by_video()
        self.id2data = {d["id"]: d for d in self.datalist}

    def __len__(self):
        return len(self.batches)

    def _get_npy_list(self, vid_id):
        path = os.path.join(self.vss_dir, vid_id)
        npy_list = [os.path.join(path, npy_name) for npy_name in os.listdir(path)]
        npy_list.sort()
        return npy_list

    def _load_npy(self, vid_id):
        if self.frm_sampling_strategy == "middle":
            npy_list = self._get_npy_list(vid_id)

            mid_idx = len(npy_list) // 2
            idx_list = [mid_idx + i for i in range(self.num_frm)]
            sampled_npy_list = [npy_list[idx] for idx in idx_list] 

            result = []
            for npy_path in sampled_npy_list:
                npy = np.load(npy_path)
                result.append(torch.tensor(npy).unsqueeze(0))
            return torch.cat(result, dim=0)
        else:
            pass

    def __getitem__(self, index):
        # skip error videos:
        batch = self.batches[index]  # one video with multiple examples
        vid_id = batch["vid_id"]
        if self.ensemble_n_clips > 1:
            # tensor (T*ensemble_n_clips, C, H, W), reshape as (T, ensemble_n_clips, C, H, W)
            # vid_frm_array = self._load_video_multi_clips_uniform(vid_id)
            pass
        else:
            vid_frm_array = self._load_npy(vid_id)  # tensor (T, C, H, W)
        batch["vid"] = vid_frm_array
        return batch

    def _prepare_batches_by_video(self):
        """create batches where each batch contains a single video with multiple text"""
        text_list = []
        for d in self.datalist:
            text_list.append(dict(
                text_str=d["txt"],
                id=d["id"],
            ))
        text_batch = dict(
            vid_id=None,
            examples=text_list,
            n_examples=len(text_list),
            ids=[d["id"] for d in text_list]
        )

        # make 1000 batches for 1000video x 1000text combinations.
        # each batch contains 1video x 1000text
        batches = []
        for d in self.datalist:
            _batch = copy.deepcopy(text_batch)
            _batch["vid_id"] = d["vid_id"]
            batches.append(_batch)
        return batches
