import torch
import torch.nn as nn


class NER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes, fc_dropout=0.3, embed_dropout=0.5, use_glove=False):
        super(NER, self).__init__()
        self.use_glove = use_glove
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, dropout=0.2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.classifier = nn.Softmax(dim=2)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)

    def forward(self, x):
        if not self.use_glove:
            x = self.embed_dropout(self.embed(x))
        x, _ = self.lstm(x)
        x = self.fc_dropout(self.fc(x))
        return x


class NER_transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_classes, n_head, fc_dropout=0.3, embed_dropout=0.5, use_glove=False):
        super(NER_transformer, self).__init__()
        self.use_glove = use_glove
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(embed_dim, n_head, 512, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.classifier = nn.Softmax(dim=-1)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)

    def forward(self, x):
        if not self.use_glove:
            x = self.embed_dropout(self.embed(x))
        x = self.transformer(x)
        x = self.fc_dropout(self.fc(x))
        return x

class NER_bilstm_cnn(nn.Module):
    def __init__(self):
        super(NER_bilstm_cnn, self).__init__()
