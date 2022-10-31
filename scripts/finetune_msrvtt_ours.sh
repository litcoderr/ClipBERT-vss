cd /clipbert

horovodrun -np 2 python -m src.tasks.run_ours_retrieval \
    --train_batch_size 128 \
    --val_batch_size 128 \
    --config src/configs/msrvtt_ret_ours.json \
    --output_dir /clipbert/results/ours_with_pe
