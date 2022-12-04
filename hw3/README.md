## Environment
In this homework, I used `Python3.9` and `torch1.12.1`. Others are in the requirement.txt.

```shell
# If you have conda, I recommend you to build a conda environment called "adl-hw3"
conda activate adl-hw3
pip install -r requirements.txt
pip install -e tw_rouge
```

## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)

## Summarization

### Training
Please check `run_train.sh` for the details.
```shell
bash run_train.sh ${1} ${2} ${3}
```
- train_file: Path to train.jsonl ${1}
- validation_file: Path to public.jsonl ${2}
- test_file: Path to public.jsonl ${2}
- model_name_or_path: <model_name> (e.g google/mt5-small)
- output_dir: Path to save model (e.g. ./tst-summarization)
- output_file_name: Path to save prediction result (e.g. ./output.jsonl) ${3}

- Plot learning curve
https://docs.google.com/spreadsheets/d/18jZRjm-vlK_-hjpRpYC8-kmQOe29Sp33MvAh0BYs760/edit?usp=sharing

### Inference
Please check `run.sh` for the details.
```shell
bash run.sh ${1} ${2}
```
- test_file: Path to private.json
- output_file_name: Path to save prediction result (e.g. ./output.jsonl) ${2}
- [num_beams]: Number of beam search sizes (e.g 5)
- [do_sample][top_p]: threshold for the CDF (e.g 0.6)
- [do_sample][top_k]: Select first K probabilities (e.g 5)
- [temperature]: Sharpen the distribution (e.g. 0.7)
