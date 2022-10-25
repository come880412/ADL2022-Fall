from curses import savetty
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

import torch
from torch.utils.data import DataLoader

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def validation(args, model, dataset, test_loader):
    pbar = tqdm.tqdm(total=len(test_loader), ncols=0, desc="Test", unit="step")

    save_data = {
        "ground_truth_str": [],
        "predict_str": [],
        "ground_truth": [],
        "predict": []
    }
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            tokens, ids = batch["tokens"].to(args.device), batch["id"]
            tags = batch["tags"].numpy()
            batch_len = batch["batch_len"]
            
            pred = model(tokens)
            
            pred_labels = torch.argmax(pred, dim=2)
            pred_labels = pred_labels.cpu().detach().numpy()

            for i in range(len(pred_labels)):
                pred_label = pred_labels[i]
                tag = tags[i]
                pred_result = []
                label_result = []
                for j in range(batch_len[i]):
                    pred_result.append(dataset.idx2label(pred_label[j])) 
                    label_result.append(dataset.idx2label(tag[j])) 
                save_data["predict_str"].append(pred_result)
                save_data["ground_truth_str"].append(label_result)
                save_data["predict"].append(pred_label[:batch_len[i]])
                save_data["ground_truth"].append(tag[:batch_len[i]])

            pbar.update()
        pbar.close()

    return save_data

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.data_dir.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, args.fix_embedding, dataset.num_classes).to(args.device)

    model.eval()

    ckpt = torch.load(args.ckpt_dir)
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    save_data = validation(args, model, dataset, val_loader)
    predicts, ground_truths = save_data["predict"], save_data["ground_truth"]
    joint_total, joint_corr = 0, 0
    token_total, token_corr = 0, 0
    for i in range(len(predicts)):
        predict = np.array(predicts[i])
        ground_truth = np.array(ground_truths[i])
        joint_corr += 1 if np.sum(np.equal(predict, ground_truth)) == len(ground_truth) else 0
        token_corr += np.sum(np.equal(predict, ground_truth))
        joint_total += 1
        token_total += len(ground_truth)

    joint_acc = (joint_corr / joint_total) * 100
    token_acc = (token_corr / token_total) * 100
    metric = classification_report(save_data["ground_truth_str"], save_data["predict_str"], scheme=IOB2, mode="strict")

    print("Joint accuracy: {:.3f}% ({}/{}) |  Token accuracy: {:.3f}% ({}/{})".format(joint_acc, joint_corr, joint_total, 
                                                                                      token_acc, token_corr, token_total))
    
    print(metric)
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/eval.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/model_best.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=48)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--fix_embedding", type=bool, default=True)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)