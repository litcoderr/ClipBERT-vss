cd /clipbert

python -m src.tasks.run_video_retrieval \
    --do_inference 1 \
    --output_dir /clipbert/results/clipbert_msrvtt \
    --inference_split val \
    --inference_model_step 0 \
    --inference_txt_db /txt/downstream/msrvtt_retrieval/msrvtt_retrieval_val.jsonl \
    --inference_img_db /img/msrvtt \
    --inference_batch_size 64 \
    --inference_n_clips 1