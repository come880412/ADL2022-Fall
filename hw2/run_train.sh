# uer/roberta-base-chinese-extractive-qa
python run_multiple-choice.py --train_file_path ../dataset/train.json \
                                --validation_file_path ../dataset/valid.json \
                                --context_file_path ../dataset/context.json \
                                --max_seq_length 512 \
                                --model_name_or_path uer/roberta-base-chinese-extractive-qa \
                                --output_dir ./roberta-base_v2/multiple-choice \
                                --do_train \
                                --do_eval \
                                --overwrite_output_dir \
                                --num_train_epochs 3 \
                                --learning_rate 3e-5 \
                                --cache_dir ./cache \
                                --pad_to_max_length \

python run_qa.py --model_name_or_path uer/roberta-base-chinese-extractive-qa \
                 --do_train \
                 --do_eval \
                 --learning_rate 3e-5 \
                 --num_train_epochs 3 \
                 --max_seq_length 512 \
                 --output_dir ./roberta-base_v2/qa \
                 --train_file_path ../dataset/train.json \
                 --validation_file_path ../dataset/valid.json \
                 --context_file_path ../dataset/context.json \
                 --cache_dir ./cache/ \
                 --doc_stride 128 \

# ckiplab/bert-base-chinese-qa
# python run_multiple-choice.py --train_file_path ../dataset/train.json \
#                                 --validation_file_path ../dataset/valid.json \
#                                 --context_file_path ../dataset/context.json \
#                                 --max_seq_length 512 \
#                                 --model_name_or_path ckiplab/bert-base-chinese-qa \
#                                 --output_dir ./bert-base/multiple-choice \
#                                 --do_train \
#                                 --do_eval \
#                                 --overwrite_output_dir \
#                                 --num_train_epochs 3 \
#                                 --learning_rate 3e-5 \
#                                 --cache_dir ./cache \
#                                 --pad_to_max_length \

# python run_qa.py --model_name_or_path ckiplab/bert-base-chinese-qa \
#                  --do_train \
#                  --do_eval \
#                  --learning_rate 3e-5 \
#                  --num_train_epochs 3 \
#                  --max_seq_length 512 \
#                  --output_dir ./bert-base/qa \
#                  --train_file_path ../dataset/train.json \
#                  --validation_file_path ../dataset/valid.json \
#                  --context_file_path ../dataset/context.json \
#                  --cache_dir ./cache/ \
#                  --doc_stride 128 \

# from scratch
# python run_multiple-choice.py --train_file_path ../dataset/train.json \
#                                 --validation_file_path ../dataset/valid.json \
#                                 --context_file_path ../dataset/context.json \
#                                 --max_seq_length 512 \
#                                 --model_name_or_path None \
#                                 --output_dir ./bert-base_fromscratch/multiple-choice \
#                                 --do_train \
#                                 --do_eval \
#                                 --overwrite_output_dir \
#                                 --num_train_epochs 3 \
#                                 --learning_rate 3e-5 \
#                                 --cache_dir ./cache \
#                                 --pad_to_max_length \
#                                 --config_name ./bert-base/multiple-choice \
#                                 --tokenizer_name ckiplab/bert-base-chinese-qa \
#                                 --from_scratch True

# python run_qa.py --model_name_or_path None \
#                  --do_train \
#                  --do_eval \
#                  --learning_rate 3e-5 \
#                  --num_train_epochs 3 \
#                  --max_seq_length 512 \
#                  --output_dir ./bert-base_fromscratch/qa \
#                  --train_file_path ../dataset/train.json \
#                  --validation_file_path ../dataset/valid.json \
#                  --context_file_path ../dataset/context.json \
#                  --cache_dir ./cache/ \
#                  --doc_stride 128 \
#                  --config_name ./bert-base/qa \
#                  --tokenizer_name ckiplab/bert-base-chinese-qa \
#                  --from_scratch True


# Train intent cls
# python run_intent_cls.py --model_name_or_path bert-base-cased \
#                          --do_train \
#                          --do_eval \
#                          --train_file_path ../../hw1/data/intent/train.json \
#                          --validation_file_path ../../hw1/data/intent/eval.json \
#                          --max_seq_length 128 \
#                          --learning_rate 2e-5 \
#                          --num_train_epochs 3 \
#                          --output_dir ./intent_cls \
#                          --overwrite_cache

# Train slot tagging
# python run_slot.py   --model_name_or_path bert-base-uncased \
#                     --train_file_path ../../hw1/data/slot/train.json \
#                     --validation_file_path ../../hw1/data/slot/eval.json \
#                     --output_dir ./slot_tagging \
#                     --do_train \
#                     --do_eval \
#                     --text_column_name tokens \
#                     --label_column_name tags \
#                     --overwrite_cache \
#                     --learning_rate 2e-5 \
#                     --num_train_epochs 3 