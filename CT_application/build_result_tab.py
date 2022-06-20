import os
import pandas as pd
import json


if __name__ == '__main__':

    single_tissue_data = {'tissue_name': [], 'feature': [], 'LR': [], 'early_stop': [], 'type_data': [],
                          'average_train_AUC': [], 'average_val_AUC': [], 'average_test_AUC': []}
    multi_tissues_data = {'feature': [], 'LR': [], 'early_stop': [], 'type_data': [],
                          'average_train_AUC': [], 'average_val_AUC': [], 'average_test_AUC': []}
    dendro_data = {'LR': [], 'L1': [], 'DPF': [], 'embedding_size': [], 'early_stop': [], 'type_data': [],
                   'average_train_AUC': [], 'average_val_AUC': [], 'average_test_AUC': []}

    print('Single tissue experiments:')
    for tissue_dir in os.listdir(os.path.join('results', 'single_tissue_experiments')):
        print(tissue_dir)
        for exp in os.listdir(os.path.join('results', 'single_tissue_experiments', tissue_dir)):
            print(exp)
            exp = exp.split('_')
            single_tissue_data['tissue_name'].append(tissue_dir)
            single_tissue_data['feature'].append(exp[0])
            single_tissue_data['LR'].append(exp[1])
            single_tissue_data['early_stop'].append(exp[2])
            single_tissue_data['type_data'].append(exp[3][0:-5])

            with open(os.path.join('results', 'single_tissue_experiments', tissue_dir, exp), 'r') as dict_file:
                auc_dict = json.load(dict_file)

            average_train_auc = 0.0
            average_val_auc = 0.0
            average_test_auc = 0.0

            for train, val, test in zip(auc_dict['train_auc'], auc_dict['val_auc'], auc_dict['test_auc']):
                average_train_auc += train
                average_val_auc += val
                average_test_auc += test

            single_tissue_data['average_train_AUC'].append(average_train_auc / len(auc_dict['train_auc']))
            single_tissue_data['average_val_AUC'].append(average_val_auc / len(auc_dict['val_auc']))
            single_tissue_data['average_test_AUC'].append(average_test_auc / len(auc_dict['test_auc']))

    single_tissues_df = pd.DataFrame(single_tissue_data)
    print(single_tissues_df)

    print('Multi Tissue Experiments:')
    for exp in os.listdir(os.path.join('results', 'multi_tissues_experiments')):
        print(exp)
        exp = exp.split('_')
        multi_tissues_data['feature'].append(exp[0])
        multi_tissues_data['LR'].append(exp[1])
        multi_tissues_data['early_stop'].append(exp[2])
        multi_tissues_data['type_data'].append(exp[3][0:-5])

        with open(os.path.join('results', 'multi_tissues_experiments', exp), 'r') as dict_file:
            auc_dict = json.load(dict_file)

        average_train_auc = 0.0
        average_val_auc = 0.0
        average_test_auc = 0.0

        for train, val, test in zip(auc_dict['train_auc'], auc_dict['val_auc'], auc_dict['test_auc']):
            average_train_auc += train
            average_val_auc += val
            average_test_auc += test

        multi_tissues_data['average_train_AUC'].append(average_train_auc / len(auc_dict['train_auc']))
        multi_tissues_data['average_val_AUC'].append(average_val_auc / len(auc_dict['val_auc']))
        multi_tissues_data['average_test_AUC'].append(average_test_auc / len(auc_dict['test_auc']))

    multi_tissues_df = pd.DataFrame(multi_tissues_data)
    print(multi_tissues_df)

    print('Dendronet Embeddings experiments:')
    for exp in os.listdir(os.path.join('results', 'dendronet_embedding_experiments')):
        print(exp)
        exp = exp.split('_')
        dendro_data['feature'].append(exp[0])
        dendro_data['LR'].append(exp[1])
        dendro_data['DPF'].append(exp[2])
        dendro_data['L1'].append(exp[3])
        dendro_data['embedding_size'].append(exp[4])
        dendro_data['early_stop'].append(exp[5])
        dendro_data['type_data'].append(exp[6][0:-5])

        average_train_auc = 0.0
        average_val_auc = 0.0
        average_test_auc = 0.0

        for train, val, test in zip(auc_dict['train_auc'], auc_dict['val_auc'], auc_dict['test_auc']):
            average_train_auc += train
            average_val_auc += val
            average_test_auc += test

        dendro_data['average_train_AUC'].append(average_train_auc / len(auc_dict['train_auc']))
        dendro_data['average_val_AUC'].append(average_val_auc / len(auc_dict['val_auc']))
        dendro_data['average_test_AUC'].append(average_test_auc / len(auc_dict['test_auc']))

    dendro_df = pd.DataFrame(dendro_data)
    print(dendro_df)
