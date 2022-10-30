cd /clipbert

horovodrun -np 4 python -m src.tasks.run_video_retrieval \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --config src/configs/msrvtt_ret_base_resnet50.json \
    --output_dir /clipbert/results/clipbert_msrvtt
