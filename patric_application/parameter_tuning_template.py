import os
import argparse
from build_data_tab import build_tab

parser = argparse.ArgumentParser()
parser.add_argument('--antibiotics', type=str, default=['erythromycin', 'betalactam'], metavar='ANT')
parser.add_argument('--genome-lineage', type=str, default='data_files/genome_lineage.csv')
#parser.add_argument('--label-file', type=str, default='data_files/erythromycin_firmicutes_sample.csv')
parser.add_argument('--dpfs', type=float, default=[0.001, 0.01, 0.1, 1.0], help='Default is [0.001, 0.01, 0.1, 1.0]')
parser.add_argument('--lrs', type=float, default=[0.01, 0.001, 0.0001], help='Default is [0.01, 0.001, 0.0001]')
parser.add_argument('--l1s', type=float, default=[0.0, 0.01, 0.1, 1.0], help='Default is [0.0, 0.01, 0.1, 1.0]')
parser.add_argument('--epochs', type=int, default=[3], help='Default is 10')
parser.add_argument('--seeds', type=int, default=[0,1,2,3,4], help='Default is [0,1,2,3,4]')
args = parser.parse_args()
if __name__ == "__main__":

    for antibiotic in args.antibiotics:

        # making some adjustments based on results from previous tuning
        dpf_list = args.dpfs
        lr_list = args.lrs
        l1_list = args.l1s
        epoch_list = args.epochs

        #tree_path = ''
        #feature_csv_path = ''

        for dpf in dpf_list:
            for lr in lr_list:
                for epoch in epoch_list:
                    for l1 in l1_list:
                        output_dir = 'data_files/patric_tuning/' + antibiotic + '_firmicutes_samples' \
                                    + '_' + str(dpf) + '_' + str(lr) + '_' + str(l1)
                        command = 'python experiment2.py --epochs ' + str(epoch) + ' --dpf ' + str(dpf) \
                                    + ' --lr ' + str(lr) + ' --output-dir ' + output_dir \
                                    + ' --l1 ' + str(l1) + ' --lineage-path ' + str(args.genome_lineage) \
                                    + ' --label-file ' + 'data_files/' + antibiotic + '_firmicutes_samples.csv'#+ ARGS FOR DATA PATH

                        os.system(command)

    df = build_tab(args.seeds)

    print(df)