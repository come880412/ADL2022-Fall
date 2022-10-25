import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import os
import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier, SeqClassifier_transformer
from utils import Vocab


def Testing(args, model, dataset, test_loader):
    pbar = tqdm.tqdm(total=len(test_loader), ncols=0, desc="Test", unit="step")

    save_data = [["id", "intent"]]
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            text, ids = batch["text"].to(args.device), batch["id"]
            
            pred = model(text)
            
            pred_labels = torch.argmax(pred, dim=1)
            pred_labels = pred_labels.cpu().detach().numpy()

            for i in range(len(pred_labels)):
                pred_label = dataset.idx2label(pred_labels[i])
                id = ids[i]
                save_data.append([id, pred_label])

            pbar.update()
        pbar.close()

    return save_data

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, args.fix_embedding, dataset.num_classes).to(args.device)
    # model = SeqClassifier_transformer(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, args.fix_embedding, dataset.num_classes).to(args.device)

    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    save_data = Testing(args, model, dataset, test_loader)
    
    # TODO: write prediction to file (args.pred_file)
    print("---------- Save data ----------")
    np.savetxt(args.pred_file,  save_data, fmt='%s', delimiter=',')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

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
