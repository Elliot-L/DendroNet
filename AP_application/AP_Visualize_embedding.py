import matplotlib.pyplot as plt
import json
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--DPF', type=float, default=0.00001)
    parser.add_argument('--L1', type=float, default=0.01)
    parser.add_argument('--balanced', default=True, action='store_true')
    parser.add_argument('--unbalanced', dest='balanced', action='store_false')
    parser.add_argument('--embedding-size', type=int, default=2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--early-stopping', type=int, default=3)
    parser.add_argument('--num-epochs', type=int, default=100)

    args = parser.parse_args()

    LR = args.LR
    DPF = args.DPF
    L1 = args.L1
    balanced = args.balanced
    embedding_size = args.embedding_size
    early_stop = args.early_stopping

    if balanced:
        data_type = '_balanced'
    else:
        data_type = '_unbalanced'

    exp_name = str(LR) + '_' + str(DPF) + '_' + str(L1) + '_' \
               + str(embedding_size) + '_' + str(early_stop) + data_type

    with open(os.path.join('results', 'dendronet_embedding_experiments', exp_name, 'baselineEmbedding.json'), 'r') as emb_file:
        embedding_dict = json.load(emb_file)

    x = []
    y = []

    for tissue, emb in embedding_dict.items():
        x.append(emb[0][0])
        y.append(emb[0][1])

    plt.scatter(x, y)

    for i, tissue in enumerate(embedding_dict.keys()):
        plt.annotate(tissue, (x[i], y[i]))

    plt.show()
