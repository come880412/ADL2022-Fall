# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_slot.py --data_dir "${1}" --ckpt_dir ./model_slot.pt --pred_file "${2}"