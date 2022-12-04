import json

import torch
import torch.nn.functional as F

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_path = "./dataset/public.jsonl"
    data = load_jsonl(data_path)
    print(data[0]["id"])
    print(data[-1]["id"])