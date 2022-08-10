import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator, GloVe
from utils import getData, iterator, train_and_eval, predict, Data

# Data
data = pd.read_csv("../data/ner_datasetreference.csv", encoding='latin1')
sentences, labels = getData(data)
split_idx = int(0.9 * len(sentences))
train, train_labels, test, test_labels = sentences[:split_idx], labels[:split_idx], sentences[split_idx:], labels[split_idx:]
vocab = build_vocab_from_iterator(iterator(train), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])
labels_dict = build_vocab_from_iterator(iterator(train_labels), specials=["<pad>"])
glove_vocab = GloVe(name='42B', dim=300)
use_glove = False
max_len = max([len(s) for s in sentences])
train_pipeline = Data(train, train_labels, max_len, vocab, glove_vocab, labels_dict, use_glove=use_glove)
test_pipeline = Data(test, test_labels, max_len, vocab, glove_vocab, labels_dict, use_glove=use_glove)
train_loader = DataLoader(train_pipeline, batch_size=64, shuffle=True)
test_loader = DataLoader(test_pipeline)

# Hyperparameters
vocab_size = len(vocab)
tag_size = len(labels_dict)
embed_size = 300
num_layers = 5
hidden_size = 64
n_epochs = 20
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
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

# Training
model = NER(vocab_size, embed_size, hidden_size, num_layers, tag_size, use_glove=use_glove).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(n_epochs):
    train_and_eval(model, train_loader, test_loader, optimizer, criterion, device, epoch)

# Testing
print(predict(model, "A spokesman says he expects the Tibetan leader to return", max_len, vocab, glove_vocab, labels_dict, use_glove, device))

# Save Model
torch.save(model.state_dict(), 'saved.pth')