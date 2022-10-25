## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip instsall -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent classification
### Training (from scratch)
```shell
python train_intent.py --data_dir <data_dir> --cache_dir <cache_dir> --ckpt_dir <ckpt_dir> --max_len 48 --hidden_size 512 --num_layers 2 --dropout 0.6 --fix_embedding True --bidirectional True --lr 1e-3 --weight_decay 1e-3 --grad_clip 5. --batch_size 64
```
- data_dir: Path to the dataset
- cache_dir: Path to the processed data
- ckpt_dir Path to save model
- max_len: The maximum sequence lengths
- hidden_size: The hidden state dimension
- num_layers: Number of layers
- dropout: Dropout rate
- fix_embedding: Whether to fix embedding layer during training
- bidirectional: Whether to use bidirectional RNN
- lr: Learning rate
- weight_decay: Penalty for l2-regularization
- grad_clip: Gradient clipping
- batch_size: Batch size

### Inference on validation set
```shell
python train_intent.py --data_dir <data_dir> --cache_dir <cache_dir> --load <model_path> --val_only True
```
- load: Path to trained model
- val_only: Conduct validation only

### Inference on testing set (kaggle)
```shell
bash intent_cls.sh ${1} ${2}
```
- "${1}" : Path to testing file
- "${2}" : Path to predicted file

## Slot Tagging
### Training (from scratch)
```shell
python train_slot.py --data_dir <data_dir> --cache_dir <cache_dir> --ckpt_dir <ckpt_dir> --max_len 48 --hidden_size 512 --num_layers 2 --dropout 0.6 --fix_embedding True --bidirectional True --lr 1e-3 --weight_decay 1e-3 --grad_clip 5. --batch_size 64
```
- data_dir: Path to the dataset
- cache_dir: Path to the processed data
- ckpt_dir Path to save model
- max_len: The maximum sequence lengths
- hidden_size: The hidden state dimension
- num_layers: Number of layers
- dropout: Dropout rate
- fix_embedding: Whether to fix embedding layer during training
- bidirectional: Whether to use bidirectional RNN
- lr: Learning rate
- weight_decay: Penalty for l2-regularization
- grad_clip: Gradient clipping
- batch_size: Batch size

### Inference on validation set
```shell
python train_slot.py --data_dir <data_dir> --cache_dir <cache_dir> --load <model_path> --val_only True
```
- load: Path to trained model
- val_only: Conduct validation only

### Inference on testing set (kaggle)
```shell
bash slot_tag.sh ${1} ${2}
```
- "${1}" : Path to testing file
- "${2}" : Path to predicted file