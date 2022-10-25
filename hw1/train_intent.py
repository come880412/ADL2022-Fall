import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import tqdm

from dataset import SeqClsDataset
from utils import Vocab, LinearWarmupCosineAnnealingLR
from model import SeqClassifier, SeqClassifier_transformer

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def Validation(args, model, valid_loader, criterion, epoch):
    model.eval()
    pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Val", unit="step")

    total_loss = 0.
    total_data, correct_data = 0, 0
    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            text, label = batch["text"].to(args.device), batch["label"].to(args.device)
            
            pred = model(text)
            loss = criterion(pred, label)
            
            pred_label = torch.argmax(pred, dim=1)
            pred_label = pred_label.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            correct_data += np.sum(np.equal(pred_label, label))
            total_data += len(text)
            total_loss += loss.item() * len(text)
            val_acc = (correct_data / total_data) * 100

            pbar.set_postfix(
                loss=f"{total_loss / total_data:.4f}",
                Accuracy=f"{val_acc:.3f}%",
            )
            pbar.update()
        pbar.close()
        val_loss = total_loss / total_data
    
    return val_loss, val_acc

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=datasets["train"].collate_fn)
    valid_loader = DataLoader(datasets["eval"], batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=datasets["eval"].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, args.fix_embedding, datasets["train"].num_classes).to(args.device)
    # model = SeqClassifier_transformer(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, args.fix_embedding, datasets["train"].num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    # iter_load = iter(train_loader)
    # data_ = iter_load.next()
    # text = data_["text"].to(args.device)
    # pred = model(text)d
    # print(pred.shape)

    # TODO: init optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, args.warmup_epochs, args.num_epochs, args.warmup_start_lr, eta_min=1e-6)

    best_acc = 0.
    patience = 0
    if not args.val_only:
        for epoch in range(args.num_epochs):
            model.train()
            # TODO: Training loop - iterate over train dataloader and update model weights
            lr = optimizer.param_groups[0]['lr']
            pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc=f"Train[{epoch}/{args.num_epochs}]", unit="step")
            
            total_loss = 0.
            total_data, correct_data = 0, 0
            for idx, batch in enumerate(train_loader):
                text, label = batch["text"].to(args.device), batch["label"].to(args.device)

                optimizer.zero_grad()
                pred = model(text)
                loss = criterion(pred, label)
                loss.backward()

                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

                pred_label = torch.argmax(pred, dim=1)
                pred_label = pred_label.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                correct_data += np.sum(np.equal(pred_label, label))
                total_data += len(text)
                total_loss += loss.item() * len(text)
                train_acc = (correct_data / total_data) * 100

                pbar.set_postfix(
                    loss=f"{total_loss / total_data:.4f}",
                    Accuracy=f"{train_acc:.3f}%",
                    lr=f"{lr:.6f}",
                )
                pbar.update()
            pbar.close()
            # TODO: Evaluation loop - calculate accuracy and save model weights
            val_loss, val_acc = Validation(args, model, valid_loader, criterion, epoch)

            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "model_best.pt"))
                print("Save model!!")
                patience = 0
            else:
                patience += 1
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "model_last.pt"))
            print(f"Best epoch[{best_epoch}/{args.num_epochs}] |  val_acc: {best_acc:.3f}%")
            if patience >= args.early_stop:
                print("Early stop!!")
                break;
            
            scheduler.step()
    else:
        val_loss, val_acc = Validation(args, model, valid_loader, criterion, epoch=args.num_epochs)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument(
        "--load",
        help="Directory to load the model file.",
        default="",
    )

    # data
    parser.add_argument("--max_len", type=int, default=48)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--fix_embedding", type=bool, default=True)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument('--grad_clip', default = 5., type=float)

    # scheduler
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--early_stop", type=int, default=7)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--val_only", type=bool, default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
