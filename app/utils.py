import torch
from torch.utils.data import Dataset


def getData(data):
    data = data.fillna(method = 'ffill')
    data = data.drop(['POS'], axis = 1)
    agg_function = lambda s: [(w,t) for w,t in zip(s["Word"].values.tolist(),
                                                        s["Tag"].values.tolist())]
    group = data.groupby('Sentence #').apply(agg_function)
    sentence = [s for s in group]
    return [[word[0] for word in s] for s in sentence], [[lab[1] for lab in s] for s in sentence]

def iterator(ss):
    for s in ss:
        yield s


def train_and_eval(model, train, test, optimizer, criterion, device, epoch):
    epoch_loss = 0
    acc = 0
    count = 0

    model.train()
    for x, y, pad_idx in train:
        x = x.to(device=device)
        y = y.to(device=device)
        optimizer.zero_grad()
        predictions = model(x)
        predictions = predictions.view(-1, predictions.shape[-1])
        y = y.view(-1)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'training loss of epoch {epoch + 1} is {(epoch_loss / len(train)):.5f}')

    model.eval()
    with torch.inference_mode():
        for x, y, pad_idx in test:
            x = x.to(device=device)
            y = y.to(device=device).float().squeeze(1)
            pad_idx = pad_idx[0]
            count += pad_idx
            predictions = torch.argmax(model(x), 2)
            correct = torch.sum((predictions[0, :pad_idx] == y[0, :pad_idx]))
            acc += correct
    print(f'test acc after epoch {epoch + 1} is {(acc / count):.5f}')


def predict(model, sentence, max_len, vocab, glove_vocab, labels_dict, use_glove, device):
    model.eval()
    with torch.inference_mode():
        temp = Data([sentence.split()], None, max_len, vocab, glove_vocab, labels_dict, use_glove)
        for x, pad_idx in temp:
            x = x.to(device=device)
            if use_glove:
                x = x.reshape((1, x.shape[0], x.shape[1]))
            else:
                x = x.reshape((1, -1))
            predictions = torch.argmax(model(x), 2).squeeze().tolist()
        return labels_dict.lookup_tokens(predictions)[:pad_idx]


class Data(Dataset):
    def __init__(self, d, l, m_len, vocab, glove_vocab, labels_dict, use_glove=False):
        self.x_train = []
        self.y_train = []
        self.pad_idx = []
        self.l = l
        for i in range(len(d)):
            padding = ["<pad>"] * (m_len - len(vocab(d[i])))
            self.pad_idx.append(len(vocab(d[i])))
            new_x = d[i] + padding
            if use_glove:
                self.x_train.append(glove_vocab.get_vecs_by_tokens(new_x))
            else:
                self.x_train.append(vocab(new_x))
            if l:
                new_y = l[i] + padding
                self.y_train.append(labels_dict(new_y))
            else:
                pass
        if use_glove:
            self.x_train = torch.stack(self.x_train)
        else:
            self.x_train = torch.tensor(self.x_train)
        if l:
            self.y_train = torch.tensor(self.y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        if self.l:
            return self.x_train[idx], self.y_train[idx], self.pad_idx[idx]
        else:
            return self.x_train[idx], self.pad_idx[idx]
