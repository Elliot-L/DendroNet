import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--antibiotics', type=str, default=['erythromycin', 'betalactam'], metavar='ANT')
parser.add_argument('--genome-lineage', type=str, default='data_files/genome_lineage.csv')
#parser.add_argument('--label-file', type=str, default='data_files/erythromycin_firmicutes_sample.csv')
args = parser.parse_args()

for antibiotic in args.antibiotics:
    # making some adjustments based on results from previous tuning

    dpf_list = [0.001, 0.01, 0.1, 1.0]
    lr_list = [0.01, 0.001, 0.0001]
    l1_list = [0.0, 0.01, 0.1, 1.0]
    epoch_list = [10]

    tree_path = ''
    feature_csv_path = ''

    if __name__ == "__main__":
        for dpf in dpf_list:
            for lr in lr_list:
                for epoch in epoch_list:
                    for l1 in l1_list:
                        output_dir = '\patric_tuning/' + 'data_files/' + antibiotic + '_firmicutes_sample.csv' \
                                     + '_' + str(dpf) + '_' + str(lr) + '_' + str(l1)
                        command = 'python3 experiment2.py --epochs ' + str(epoch) + ' --dpf ' + str(dpf) \
                                  + ' --lr ' + str(lr) + ' --output-dir ' + output_dir \
                                  + ' --l1 ' + str(l1) + ' --lineage-path ' + str(args.genome_lineage) \
                                  + ' --label-file ' + 'data_files/' + antibiotic + '_firmicutes_sample.csv'#+ ARGS FOR DATA PATH

                        os.system(command)
