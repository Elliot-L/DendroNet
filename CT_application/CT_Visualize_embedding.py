import matplotlib.pyplot as plt
import json
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='active')
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--DPF', type=float, default=0.01)
    parser.add_argument('--L1', type=float, default=0.01)
    parser.add_argument('--EL', type=float, default=0.0)
    parser.add_argument('--balanced', default=True, action='store_true')
    parser.add_argument('--unbalanced', dest='balanced', action='store_false')
    parser.add_argument('--embedding-size', type=int, default=2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--early-stopping', type=int, default=3)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='dendronet')

    args = parser.parse_args()

    feature = args.feature
    LR = args.LR
    DPF = args.DPF
    L1 = args.L1
    EL = args.EL
    balanced = args.balanced
    embedding_size = args.embedding_size
    early_stop = args.early_stopping
    model = args.model

    if balanced:
        data_type = '_balanced'
    else:
        data_type = '_unbalanced'

    if model == 'dendronet':
        exp_name = feature + '_' + str(LR) + '_' + str(DPF) + '_' + str(L1) + '_' \
                        + str(embedding_size) + '_' + str(early_stop) + data_type

        with open(os.path.join('results', 'dendronet_embedding_experiments', exp_name, 'baselineEmbedding.json'), 'r') as emb_file:
            embedding_dict = json.load(emb_file)

    elif model == 'baseline':
        exp_name = feature + '_' + str(LR) + '_' + str(EL) + '_' \
                   + str(embedding_size) + '_' + str(early_stop) + data_type

        with open(os.path.join('results', 'baseline_embedding_experiments', exp_name, 'baselineEmbedding.json'),
                  'r') as emb_file:
            embedding_dict = json.load(emb_file)

    elif model == 'best':

    """
    Tissue categories:
        - sexual (black)
        - gland (blue)
        - muscle (yellow)
        - skin/epithelium (green)
        - blood vessels (red)
        - immune (grey)
        - digestive (orange)
        - nerve (purple)
        - respiratory (pink)
    """

    sexual = ['uterus', 'ovary', 'testis']
    glands = ['adrenal_gland', 'body_of_pancreas', 'prostate_gland',
             'thyroid_gland', 'right_lobe_of_liver']
    muscles = ['heart_left_ventricle',
              'gastroesophageal_sphincter', 'right_atrium_auricular_region',
              'vagina', 'gastrocnemius_medialis']
    skin = ['breast_epithelium', 'esophagus_squamous_epithelium',
            'suprapubic_skin']
    blood_vessels = ['ascending_aorta', 'coronary_artery',
                     'thoracic_aorta', 'tibial_artery']
    immune = ['spleen', 'Peyers_patch']
    digestive = ['transverse_colon', 'sigmoid_colon', 'stomach', 'esophagus_muscularis_mucosa']
    nerve = ['tibial_nerve']
    respiratory = ['upper_lobe_of_left_lung']

    for tissue, emb in embedding_dict.items():
        if tissue in sexual:
            plt.scatter(emb[0][0], emb[0][1], color='black')
        elif tissue in glands:
            plt.scatter(emb[0][0], emb[0][1], color='blue')
        elif tissue in muscles:
            plt.scatter(emb[0][0], emb[0][1], color='yellow')
        elif tissue in skin:
            plt.scatter(emb[0][0], emb[0][1], color='green')
        elif tissue in blood_vessels:
            plt.scatter(emb[0][0], emb[0][1], color='red')
        elif tissue in immune:
            plt.scatter(emb[0][0], emb[0][1], color='grey')
        elif tissue in digestive:
            plt.scatter(emb[0][0], emb[0][1], color='orange')
        elif tissue in nerve:
            plt.scatter(emb[0][0], emb[0][1], color='purple')
        elif tissue in respiratory:
            plt.scatter(emb[0][0], emb[0][1], color='pink')
        else:
            print(tissue)

    if model == 'dendronet':
        plt.title('Dendronet: DPF ' + str(DPF) + ' L1 ' + str(L1) + ' LR ' + str(LR))
    elif model == 'baseline':
        plt.title('Baseline: EL ' + str(EL) + ' LR ' + str(LR))

    plt.show()
