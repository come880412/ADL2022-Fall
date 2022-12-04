python run_summarization.py \
        --model_name_or_path "./summarization" \
        --output_dir "./summarization" \
        --do_predict \
        --test_file "${1}" \
        --source_prefix "summarize: " \
        --overwrite_output_dir \
        --per_device_eval_batch_size=4 \
        --text_column "maintext" \
        --output_file_name "${2}" \
        --predict_with_generate \
        --num_beams 5 
        # --temperature 0.7 \
        # --do_sample \
        # --top_p 0.6 \
        # --top_k 5 \
        # --summary_column "title"


# python eval.py \
#         -r "./dataset/public.jsonl" \
#         -s "${2}"
