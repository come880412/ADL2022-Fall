python run_multiple-choice.py --model_name_or_path ./roberta-base/multiple-choice \
                              --cache_dir ./cache/ \
                              --output_dir ./roberta-base/multiple-choice \
                              --context_file_path "${1}" \
                              --test_file_path "${2}" \
                              --output_file_path ./context_pred.json \
                              --do_predict \
                              --max_seq_length 512 \
                              --per_gpu_eval_batch_size 32 \
                              --pad_to_max_length

python run_qa.py --model_name_or_path ./roberta-base/qa \
                 --cache_dir ./cache/ \
                 --output_dir ./roberta-base/qa \
                 --context_file_path "${1}" \
                 --max_seq_length 512 \
                 --doc_stride 128 \
                 --per_device_eval_batch_size 48 \
                 --do_predict \
                 --test_file_path ./context_pred.json \
                 --output_file_path "${3}"

# Hckiplab/bert-base-chinese-qa
# python run_multiple-choice.py --model_name_or_path ./bert-base/multiple-choice \
#                               --cache_dir ./cache/ \
#                               --output_dir ./bert-base/multiple-choice \
#                               --context_file ../dataset/context.json \
#                               --test_file_path ../dataset/test.json \
#                               --output_file_path ./context_pred_bertbased.json \
#                               --do_predict \
#                               --max_seq_length 512 \
#                               --per_gpu_eval_batch_size 32 \
#                               --pad_to_max_length

# python run_qa.py --model_name_or_path ./bert-base/qa \
#                  --cache_dir ./cache/ \
#                  --output_dir ./bert-base/qa \
#                  --context_file ../dataset/context.json \
#                  --max_seq_length 512 \
#                  --doc_stride 128 \
#                  --per_device_eval_batch_size 48 \
#                  --do_predict \
#                  --test_file_path ./context_pred_bertbased.json \
#                  --output_file_path ./out.csv