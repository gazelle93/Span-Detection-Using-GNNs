import argparse
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from text_processing import get_target, word_tokenization
import my_onehot
from depgcn import Dependency_GCN
from depgat import Dependency_GAT

def main(args):
    cur_text = "However, he also doubts that a hacker would have much interest in the blood pressure readings you're sending to your doctor because if you are in his shoes, you'd find it difficult to make profit off that data."
    ant_span_text = "if you are in his shoes"
    con_span_text = "you'd find it difficult to make profit off that data"

    
    # One-hot Encoding
    sklearn_onehotencoder = my_onehot.build_onehot_encoding_model(args.unk_ignore)
    token2idx_dict, _ = my_onehot.init_token2idx([cur_text], args.nlp_pipeline)
    sklearn_onehotencoder.fit([[t] for t in token2idx_dict])
    tks, deps = my_onehot.get_tokens(cur_text, args.nlp_pipeline)

    dependency_list = list(set([x[1] for x in deps]))

    embeddings = my_onehot.onehot_encoding(sklearn_onehotencoder, tks)
    target, label_dict, reverse_label_dict = get_target(cur_text, ant_span_text, con_span_text, args.nlp_pipeline)

    input_dim, hidden_dim = embeddings.size()
    output_dim = len(set(target))

    if args.gnn.lower() == "depgcn":
        model = Dependency_GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dependency_list=dependency_list, num_layers=args.num_layers, dropout_rate=args.dropout_rate, reverse_case=args.reverse)
    elif args.gnn.lower() == "depgat":
        model = Dependency_GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, alpha=args.alpha, dependency_list=dependency_list, num_layers=args.num_layers, dropout_rate=args.dropout_rate)


    # Training
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.zero_grad()

    for _ in tqdm(range(args.epochs)):
        optimizer.zero_grad()

        output = model(embeddings, deps)

        loss = loss_function(output, torch.tensor(target))
        loss.backward()
        optimizer.step()

    print(loss)
    pred = [label_dict[x.item()] for x in torch.argmax(model(embeddings, deps), dim=1)]

    print("Antecedent spans")
    print("True Antecedent spans: {}".format([tks[idx] for idx, x in enumerate(target) if label_dict[x] == "A"]))
    print("Predicted Antecedent spans: {}".format([tks[idx] for idx, x in enumerate(pred) if x == "A"]))

    print("\nConsequent spans")
    print("True Consequent spans: {}".format([tks[idx] for idx, x in enumerate(target) if label_dict[x] == "C"]))
    print("Predicted Consequent spans: {}".format([tks[idx] for idx, x in enumerate(pred) if x == "C"]))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--unk_ignore", default=True, help="Ignore unknown tokens.")
    parser.add_argument("--num_layers", default=1, type=int, help="The number of lstm/bilstm layers.")
    parser.add_argument("--gnn", default="depgcn", type=str, help="Type of dependency gnn layer. (depgcn, depgat)")
    parser.add_argument("--alpha", default=0.01, type=float, help="Negative slope that controls the angle of the negative slope of LeakyReLU")
    parser.add_argument("--epochs", default=100, type=int, help="The number of epochs for training.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="Learning rate.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate.")
    parser.add_argument("--reverse", default=True, help="Applying reverse dependency cases (gov -> dep) or not.")
    args = parser.parse_args()

    main(args)
