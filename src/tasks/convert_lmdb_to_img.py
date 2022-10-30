"""
Converts msrvtt lmdb file to images
"""

import av
import io
import os
import lmdb
from src.utils.basic_utils import load_jsonl
from tqdm import tqdm


TXT_PATH = "/txt/msrvtt_retrieval"
JSONL_NAMES = [
    "train.jsonl",
    "val.jsonl",
    "test.jsonl"
]
LMDB_PATH = "/img/msrvtt/"
TARGET_FPS = 2
RESULT_PATH = "/img/msrvtt_img"
IMG_SIZE = (224, 224)

if __name__ == "__main__":
    # get all clip_name s
    clip_names = set()

    for jsonl_name in JSONL_NAMES:
        data_list = load_jsonl(os.path.join(TXT_PATH, jsonl_name))
        for data in data_list:
            clip_names.add(data["clip_name"])

    # extract img
    env = lmdb.open(LMDB_PATH, readonly=True, create=False)
    txn = env.begin(buffers=True)

    for img_id in tqdm(clip_names):
        io_stream = io.BytesIO(txn.get(str(img_id).encode("utf-8")))
        video_container = av.open(io_stream, metadata_errors="ignore")
        video = video_container.streams.video[0]
        original_fps = float(video.average_rate)

        os.mkdir(os.path.join(RESULT_PATH, img_id))

        augmented_idx = 0
        skip_rate = int(original_fps / TARGET_FPS)
        for frame in video_container.decode(video=0):
            if frame.index % skip_rate == 0:
                pil_image = frame.to_image()
                pil_iamge = pil_image.resize(IMG_SIZE)
                pil_image.save(os.path.join(RESULT_PATH, img_id, "%04d.jpg" % augmented_idx))
                augmented_idx += 1
