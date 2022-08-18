import os
import pandas as pd
import json


if __name__ == '__main__':

    single_tissue_data = {'tissue_name': [], 'feature': [], 'LR': [], 'early_stop': [], 'type_data': [],
                          'average_train_AUC': [], 'average_val_AUC': [], 'average_test_AUC': [],
                          #'average_train_loss': [], 'average_val_loss': [], 'average_test_loss': [],
                          'average_epochs': []}
    multi_tissues_data = {'feature': [], 'LR': [], 'early_stop': [], 'type_data': [],
                          'average_train_AUC': [], 'average_val_AUC': [], 'average_test_AUC': [],
                          #'average_train_loss': [], 'average_val_loss': [], 'average_test_loss': [],
                          'average_epochs': []}
    embedding_data = {'feature': [], 'LR': [], 'EL': [], 'embedding_size': [], 'early_stop': [],
                   'type_data': [], 'average_train_AUC': [], 'average_val_AUC': [], 'average_test_AUC': [],
                   #'average_train_loss': [], 'average_val_loss': [], 'average_test_loss': [],
                   'average_epochs': []}

    dendro_data = {'feature': [], 'LR': [], 'L1': [], 'DPF': [], 'embedding_size': [], 'early_stop': [],
                   'type_data': [], 'average_train_AUC': [], 'average_val_AUC': [], 'average_test_AUC': [],
                   #'average_train_loss': [], 'average_val_loss': [], 'average_test_loss': [],
                   'average_epochs': [], 'tree': []}

    print('Single tissue experiments:')
    for tissue_dir in os.listdir(os.path.join('results', 'single_tissue_experiments')):
        for exp_name in os.listdir(os.path.join('results', 'single_tissue_experiments', tissue_dir)):
            exp = exp_name.split('_')
            single_tissue_data['tissue_name'].append(tissue_dir)
            single_tissue_data['feature'].append(exp[0])
            single_tissue_data['LR'].append(exp[1])
            single_tissue_data['early_stop'].append(exp[2])
            single_tissue_data['type_data'].append(exp[3][0:-5])

            with open(os.path.join('results', 'single_tissue_experiments', tissue_dir,
                                   exp_name, 'output.json'), 'r') as dict_file:
                auc_dict = json.load(dict_file)

            average_train_auc = 0.0
            average_val_auc = 0.0
            average_test_auc = 0.0
            #average_train_loss = 0.0
            #average_val_loss = 0.0
            #average_test_loss = 0.0
            average_epochs = 0.0

            for train, val, test, epoch in zip(auc_dict['train_auc'], auc_dict['val_auc'], auc_dict['test_auc'], auc_dict['epochs']):
                average_train_auc += train
                average_val_auc += val
                average_test_auc += test
                average_epochs += epoch

            #for train, val, test in zip(auc_dict['train_loss'], auc_dict['val_loss'], auc_dict['test_loss']):
             #   average_train_loss += train
            #    average_val_loss += val
            #    average_test_loss += test

            single_tissue_data['average_train_AUC'].append(average_train_auc / len(auc_dict['train_auc']))
            single_tissue_data['average_val_AUC'].append(average_val_auc / len(auc_dict['val_auc']))
            single_tissue_data['average_test_AUC'].append(average_test_auc / len(auc_dict['test_auc']))
            #single_tissue_data['average_train_loss'].append(average_train_loss / len(auc_dict['train_loss']))
            #single_tissue_data['average_val_loss'].append(average_val_loss / len(auc_dict['val_loss']))
            #single_tissue_data['average_test_loss'].append(average_test_loss / len(auc_dict['test_loss']))
            single_tissue_data['average_epochs'].append(average_epochs / len(auc_dict['epochs']))

    single_tissues_df = pd.DataFrame(single_tissue_data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(single_tissues_df)

    print('Multi Tissue Experiments:')
    for exp_name in os.listdir(os.path.join('results', 'multi_tissues_experiments')):
        exp = exp_name.split('_')
        multi_tissues_data['feature'].append(exp[0])
        multi_tissues_data['LR'].append(exp[1])
        multi_tissues_data['early_stop'].append(exp[2])
        multi_tissues_data['type_data'].append(exp[3][0:-5])

        with open(os.path.join('results', 'multi_tissues_experiments', exp_name, 'output.json'), 'r') as dict_file:
            auc_dict = json.load(dict_file)

        average_train_auc = 0.0
        average_val_auc = 0.0
        average_test_auc = 0.0
        #average_train_loss = 0.0
        #average_val_loss = 0.0
        #average_test_loss = 0.0
        average_epochs = 0.0

        for train, val, test, epoch in zip(auc_dict['train_auc'], auc_dict['val_auc'], auc_dict['test_auc'], auc_dict['epochs']):
            average_train_auc += train
            average_val_auc += val
            average_test_auc += test
            average_epochs += epoch

        """
        for train, val, test in zip(auc_dict['train_loss'], auc_dict['val_loss'], auc_dict['test_loss']):
            average_train_loss += train
            average_val_loss += val
            average_test_loss += test
        """

        multi_tissues_data['average_train_AUC'].append(average_train_auc / len(auc_dict['train_auc']))
        multi_tissues_data['average_val_AUC'].append(average_val_auc / len(auc_dict['val_auc']))
        multi_tissues_data['average_test_AUC'].append(average_test_auc / len(auc_dict['test_auc']))
        #multi_tissues_data['average_train_loss'].append(average_train_loss / len(auc_dict['train_loss']))
        #multi_tissues_data['average_val_loss'].append(average_val_loss / len(auc_dict['val_loss']))
        #multi_tissues_data['average_test_loss'].append(average_test_loss / len(auc_dict['test_loss']))
        multi_tissues_data['average_epochs'].append(average_epochs / len(auc_dict['epochs']))

    multi_tissues_df = pd.DataFrame(multi_tissues_data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(multi_tissues_df)

    print('Baseline Embeddings experiments:')
    for exp_name in os.listdir(os.path.join('results', 'baseline_embedding_experiments')):
        exp = exp_name.split('_')
        embedding_data['feature'].append(exp[0])
        embedding_data['LR'].append(exp[1])
        embedding_data['EL'].append(exp[2])
        embedding_data['embedding_size'].append(exp[3])
        embedding_data['early_stop'].append(exp[4])
        embedding_data['type_data'].append(exp[5][0:-5])

        with open(os.path.join('results', 'baseline_embedding_experiments', exp_name,
                               'output.json'), 'r') as dict_file:
            auc_dict = json.load(dict_file)

        average_train_auc = 0.0
        average_val_auc = 0.0
        average_test_auc = 0.0
        #average_train_loss = 0.0
        #average_val_loss = 0.0
        #average_test_loss = 0.0
        average_epochs = 0.0

        for train, val, test, epoch in zip(auc_dict['train_auc'], auc_dict['val_auc'], auc_dict['test_auc'], auc_dict['epochs']):
            average_train_auc += train
            average_val_auc += val
            average_test_auc += test
            average_epochs += epoch
        """
        for train, val, test in zip(auc_dict['train_loss'], auc_dict['val_loss'], auc_dict['test_loss']):
            average_train_loss += train
            average_val_loss += val
            average_test_loss += test
        """

        embedding_data['average_train_AUC'].append(average_train_auc / len(auc_dict['train_auc']))
        embedding_data['average_val_AUC'].append(average_val_auc / len(auc_dict['val_auc']))
        embedding_data['average_test_AUC'].append(average_test_auc / len(auc_dict['test_auc']))
        #embedding_data['average_train_loss'].append(average_train_loss / len(auc_dict['train_loss']))
        #embedding_data['average_val_loss'].append(average_val_loss / len(auc_dict['val_loss']))
        #embedding_data['average_test_loss'].append(average_test_loss / len(auc_dict['test_loss']))
        embedding_data['average_epochs'].append(average_epochs / len(auc_dict['epochs']))

    embedding_df = pd.DataFrame(embedding_data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(embedding_df)

    print('Dendronet Embeddings experiments:')
    for exp_name in os.listdir(os.path.join('results', 'dendronet_embedding_experiments')):
        exp = exp_name.split('_')
        dendro_data['feature'].append(exp[0])
        dendro_data['LR'].append(exp[1])
        dendro_data['DPF'].append(exp[2])
        dendro_data['L1'].append(exp[3])
        dendro_data['embedding_size'].append(exp[4])
        dendro_data['early_stop'].append(exp[5])
        dendro_data['type_data'].append(exp[6][0:-5])
        if len(exp) == 9:
            dendro_data['tree'].append(exp[7] + '_' + exp[8])
        else:
            dendro_data['tree'].append(exp[7])

        with open(os.path.join('results', 'dendronet_embedding_experiments', exp_name,
                               'output.json'), 'r') as dict_file:
            auc_dict = json.load(dict_file)

        average_train_auc = 0.0
        average_val_auc = 0.0
        average_test_auc = 0.0
        average_train_loss = 0.0
        average_val_loss = 0.0
        average_test_loss = 0.0
        average_epochs = 0.0

        for train, val, test, epoch in zip(auc_dict['train_auc'], auc_dict['val_auc'], auc_dict['test_auc'], auc_dict['epochs']):
            average_train_auc += train
            average_val_auc += val
            average_test_auc += test
            average_epochs += epoch

        """
        for train, val, test in zip(auc_dict['train_loss'], auc_dict['val_loss'], auc_dict['test_loss']):
            average_train_loss += train
            average_val_loss += val
            average_test_loss += test
        """

        dendro_data['average_train_AUC'].append(average_train_auc / len(auc_dict['train_auc']))
        dendro_data['average_val_AUC'].append(average_val_auc / len(auc_dict['val_auc']))
        dendro_data['average_test_AUC'].append(average_test_auc / len(auc_dict['test_auc']))
        #dendro_data['average_train_loss'].append(average_train_loss / len(auc_dict['train_loss']))
        #dendro_data['average_val_loss'].append(average_val_loss / len(auc_dict['val_loss']))
        #dendro_data['average_test_loss'].append(average_test_loss / len(auc_dict['test_loss']))
        dendro_data['average_epochs'].append(average_epochs / len(auc_dict['epochs']))

    dendro_df = pd.DataFrame(dendro_data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dendro_df)

    best = 0
    pos = 0
    for row in range(multi_tissues_df.shape[0]):
        if best < multi_tissues_df.loc[row, 'average_test_AUC']:
            best = multi_tissues_df.loc[row, 'average_test_AUC']
            pos = row

    print('Best for multi tissue baseline')
    print(multi_tissues_df.loc[pos, :])

    best = 0
    pos = 0
    for row in range(embedding_df.shape[0]):
        if best < embedding_df.loc[row, 'average_test_AUC']:
            best = embedding_df.loc[row, 'average_test_AUC']
            pos = row

    print('Best for embedding baseline')
    print(embedding_df.loc[pos, :])

    trees_used = list(set(dendro_df.loc[:, 'tree']))

    for tree in trees_used:
        best = 0
        pos = 0
        for row in range(dendro_df.shape[0]):
            if dendro_df.loc[row, 'tree'] == tree:
                if best < dendro_df.loc[row, 'average_test_AUC']:
                    best = dendro_df.loc[row, 'average_test_AUC']
                    pos = row

        print('Best for dendronet on ' + tree + ' tree: ')
        print(dendro_df.loc[pos, :])
