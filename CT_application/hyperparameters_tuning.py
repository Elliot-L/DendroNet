import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='active')
    parser.add_argument('--single_tissues', type=str, nargs='+', default=[])
    parser.add_argument('--LRs', type=float, nargs='+', default=[0.1, 0.01, 0.001])
    parser.add_argument('--L1s', type=float, nargs='+', default=[0.1, 0.01, 0.001])
    parser.add_argument('--DPFs', type=float, nargs='+', default=[0.1, 0.01, 0.001])
    parser.add_argument('--embedding-sizes', type=int, nargs='+', default=[10])  # [3, 5, 10])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--GPU', default=True, action='store_true')
    parser.add_argument('--CPU', dest='GPU', action='store_false')
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    # parser.add_argument('--whole-dataset', type=bool, choices=[True, False], default=False)
    parser.add_argument('--balanced', default=True, action='store_true')
    parser.add_argument('--unbalanced', dest='balanced', action='store_false')
    parser.add_argument('--force-train', type=bool, default=False, choices=[True, False],
                        help='train even if result file exists')
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--models-to-train', type=str, default='smd', help='s: single, m:multiple, d:dendronet')
    parser.add_argument('--early-stopping', type=int, default=3, help='number of epoches after which, if no'
                                                                      + 'improvement is observed on the AUC of '
                                                                      + 'the validation set, training is stopped.')
    args = parser.parse_args()

    feature = args.feature
    dpf_list = args.DPFs
    lr_list = args.LRs
    l1_list = args.L1s
    embedding_sizes = args.embedding_sizes
    early_stop = args.early_stopping
    epochs = args.num_epochs
    balanced = args.balanced
    force_train = args.force_train
    models_to_train = args.models_to_train
    USE_CUDA =args.GPU
    single_tissues = args.single_tissues

    if not single_tissues:
        for tissue_matrix in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
            tissue_name = tissue_matrix[0:-29]
            single_tissues.append(tissue_name)

    seeds_str = ''
    for s in args.seeds:
        seeds_str += ' '
        seeds_str += str(s)

    if balanced:
        type_data = '_balanced'
    else:
        type_data = '_unbalanced'

    if 's' in models_to_train:
        print('Baseline 1: Tissue Specific models')
        for LR in lr_list:
            for tissue_matrix in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
                tissue_name = tissue_matrix[0:-29]
                if tissue_name in single_tissues:
                    tissue_file = os.path.join('results', 'single_tissue_experiments', tissue_name,
                                                        feature + '_' + str(LR) + '_' + str(early_stop) + type_data)
                    print(tissue_file)
                    if not os.path.isfile(tissue_file or force_train):
                        command = 'python CT_specific_conv_experiment.py' \
                                  + ' --ct ' + tissue_name \
                                  + ' --feature ' + feature \
                                  + ' --LR ' + str(LR) \
                                  + ' --early-stopping ' + str(early_stop) \
                                  + ' --num-epochs ' + str(epochs) \
                                  + ' --seeds' + seeds_str \

                        if USE_CUDA:
                            command += ' --GPU'
                        else:
                            command += ' --CPU'

                        if balanced:
                            command += ' --balanced'
                        else:
                            command += ' --unbalanced'

                        os.system(command)

    if 'm' in models_to_train:
        print('Baseline 2: Multi Tissue model')
        for LR in lr_list:
            multi_file = os.path.join('results', 'multi_tissue_experiments',
                                      feature + '_' + str(LR) + '_' + str(early_stop) + type_data)
            print(multi_file)
            if not os.path.isfile(multi_file or force_train):
                command = 'python MultipleCT_conv_exp.py' \
                          + ' --feature ' + feature \
                          + ' --LR ' + str(LR) \
                          + ' --early-stopping ' + str(early_stop) \
                          + ' --num-epochs ' + str(epochs) \
                          + ' --seeds' + seeds_str \

                if USE_CUDA:
                    command += ' --GPU'
                else:
                    command += ' --CPU'

                if balanced:
                    command += ' --balanced'
                else:
                    command += ' --unbalanced'

                os.system(command)

    if 'd' in models_to_train:
        print('Model that uses the Dendronet tissue embedding')
        for LR in lr_list:
            for L1 in l1_list:
                for DPF in dpf_list:
                    for emb_size in embedding_sizes:
                        dendro_file = os.path.join('results', 'dendronet_embedding_experiments',
                                                   feature + '_' + str(LR) + '_' + str(DPF) + '_' + str(L1)
                                                   + '_' + str(emb_size) + '_' + str(early_stop) + type_data)
                        print(dendro_file)
                        if not os.path.isfile(dendro_file or force_train):
                            command = 'python DendroEmbeddingExperiment.py' \
                                      + ' --feature ' + feature \
                                      + ' --LR ' + str(LR) \
                                      + ' --DPF ' + str(DPF) \
                                      + ' --L1 ' + str(L1) \
                                      + ' --embedding-size ' + str(emb_size) \
                                      + ' --early-stopping ' + str(early_stop) \
                                      + ' --num-epochs ' + str(epochs) \
                                      + ' --seeds' + seeds_str

                            if USE_CUDA:
                                command += ' --GPU'
                            else:
                                command += ' --CPU'

                            if balanced:
                                command += ' --balanced'
                            else:
                                command += ' --unbalanced'

                            os.system(command)













