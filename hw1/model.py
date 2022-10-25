import imp
from typing import Dict

import torch
import torch.nn as nn

from utils import Smoother


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        fix_embedding: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.fix_embedding = fix_embedding
        self.embedding = 300
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=self.fix_embedding)
        # TODO: model architecture

        self.gru = nn.GRU(self.embedding, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.classifier = nn.Sequential(
                                nn.Dropout(self.dropout),
                                nn.Linear(self.hidden_size * 2, self.num_class)
                            )
        else:
            self.classifier = nn.Sequential(
                                nn.Dropout(self.dropout),
                                nn.Linear(self.hidden_size, self.num_class)
                            )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        inputs = self.embed(batch)
        x, _ = self.gru(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用最後一層的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


class SeqClassifier_transformer(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        fix_embedding: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier_transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.fix_embedding = fix_embedding
        self.embedding = 300
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=self.fix_embedding)
        # TODO: model architecture

        self.encoder_layer = Smoother(
            d_model=self.embedding, dim_feedforward=self.hidden_size, nhead=2, dropout=self.dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.classifier = nn.Sequential(
                            nn.Dropout(self.dropout),
                            nn.Linear(self.embedding, self.num_class)
                        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        inputs = self.embed(batch)
        # out: (length, batch size, embedding)
        out = inputs.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, embedding).
        out = self.encoder(out)
        # out: (batch size, length, embedding)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, num_classes)
        out = self.classifier(stats)
        return out


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        inputs = self.embed(batch)
        x, _ = self.gru(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        x = self.classifier(x)
        return x


class SeqTagger_transformer(SeqClassifier_transformer):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        inputs = self.embed(batch)
        # out: (length, batch size, embedding)
        out = inputs.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, embedding).
        out = self.encoder(out)
        # out: (batch size, length, embedding)
        out = out.transpose(0, 1)

        # out: (batch, length, num_classes)
        out = self.classifier(out)
        return out
