cd /clipbert

python -m src.tasks.run_video_retrieval \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --config src/configs/msrvtt_ret_base_resnet50.json \
    --output_dir /clipbert/results/clipbert_msrvtt