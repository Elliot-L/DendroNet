import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='active')
    parser.add_argument('--LRs', type=float, nargs='+', default=[0.1, 0.01, 0.001])
    parser.add_argument('--L1s', type=float, nargs='+', default=[0.1, 0.01, 0.001])
    parser.add_argument('--DPFs', type=float, nargs='+', default=[0.1, 0.01, 0.001])
    parser.add_argument('--embedding-sizes', type=int, nargs='+', default=[3, 5, 10])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--USE-CUDA', type=bool, default=False)
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    parser.add_argument('--whole-dataset', type=bool, default=False)
    parser.add_argument('--num-epoches', type=int, default=100)
    parser.add_argument('--model-to-train', type=str, default='smd', help='s: single, m:multiple, d:dendronet')
    parser.add_argument('--early-stopping', type=int, default=3, help='number of epoches after which, if no'
                                                                      + 'improvement is observed on the AUC of '
                                                                      + 'the validation set, training is stopped.')
    args = parser.parse_args()

    feature = args.feature
    dpf_list = args.DPFs
    lr_list = args.LRs
    l1_list = args.L1s
    embedding_size = args.embedding_sizes
    early_stop =args.early_stopping
    epochs = args.num_epoches
    whole_dataset = args.whole_dataset

    seeds_str = ''
    for s in args.seeds:
        seeds_str += ' '
        seeds_str += str(s)

    if whole_dataset:
        type_data = '_unbalanced'
    else:
        type_data = '_balanced'

    print('Baseline 1: Tissue Specific models')
    for LR in lr_list:
        for tissue_matrix in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
            tissue_name = tissue_matrix[0:-29]
            tissue_file = os.path.join('results', 'single_tissue_experiments', tissue_name,
                                                feature + '_' + str(LR) + '_' + str(early_stop) + type_data)
            print(tissue_file)
            if not os.path.isfile(tissue_file):
                command = 'python CT_specific_conv_experiment.py'
                          + ' --ct ' + tissue_name
                          + ' --feature ' + feature
                          + ' --LR ' + LR
                          + ' --early-stopping ' + early_stop
                          + ' --num-epoches ' + epochs
                          + ' --seeds' + seeds_str
                          + ' --whole-dataset ' + whole_dataset

                os.system(command)

    print('Baseline 2: Multi Tissue model')
    for LR in lr_list:
        multi_file = os.path.join('results', 'single_tissue_experiments', tissue_name,
                                                feature + '_' + str(LR) + '_' + str(early_stop) + type_data)
        print(multi_file)
        if not os.path.isfile(multi_file):
            command = 'python MultipleCT_conv_exp.py'
            + ' --feature ' + feature
            + ' --LR ' + LR
            + ' --early-stopping ' + early_stop
            + ' --num-epoches ' + epochs
            + ' --seeds' + seeds_str
            + ' --whole-dataset ' + whole_dataset

            os.system(command)

    print('Model that uses the Dendronet tissue embedding')
    for LR in lr_list:
        for L1 in l1_list:
            for DPF in dpf_list:
                for emb_size in embedding_size:












