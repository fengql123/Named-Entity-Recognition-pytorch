from flask import Flask, request, jsonify
import torch
from model import NER
from utils import predict

vocab = torch.load("../vocabs/vocab_obj.pth")
labels_dict = torch.load("../vocabs/labels_dict.pth")
glove_vocab = torch.load("../vocabs/glove_vocab.pth")
max_len = 104
vocab_size = len(vocab)
tag_size = len(labels_dict)
embed_size = 300
num_layers = 5
hidden_size = 64
use_glove = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NER(vocab_size, embed_size, hidden_size, num_layers, tag_size, use_glove=use_glove).to(device=device)
model.load_state_dict(torch.load("./saved_no_glove.pth"))

app = Flask(__name__)


@app.route("/classify", methods=["POST"])
def classify():
    if request.method == 'POST':
        sentence = request.json["sentence"]
    if not sentence:
        return jsonify({'error': 'sentence is null'})
    else:
        prediction = ' '.join(predict(model, sentence, max_len, vocab, glove_vocab, labels_dict, use_glove, device))
        data = {"prediction": prediction}
        return jsonify(data)



