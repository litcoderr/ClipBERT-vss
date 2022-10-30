cd /clipbert

horovodrun -np 2 python -m src.tasks.run_ours_retrieval \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --config src/configs/msrvtt_ret_ours.json \
    --output_dir /clipbert/results/ours
