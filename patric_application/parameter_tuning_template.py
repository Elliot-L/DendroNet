import os
import argparse
from build_data_tab import build_tab
import json

parser = argparse.ArgumentParser()
parser.add_argument('--antibiotic', type=str, default='erythromycin', metavar='ANT')
parser.add_argument('--group', type=str, default='firmicutes', metavar='GR')
parser.add_argument('--genome-lineage', type=str, default=os.path.join('data_files', 'genome_lineage.csv'))
parser.add_argument('--dpfs', type=float, default=[0.001, 0.01, 0.1, 1.0], help='Default is [0.001, 0.01, 0.1, 1.0]')
parser.add_argument('--lrs', type=float, default=[0.01, 0.001, 0.0001], help='Default is [0.01, 0.001, 0.0001]')
parser.add_argument('--l1s', type=float, default=[0.0, 0.01, 0.1, 1.0], help='Default is [0.0, 0.01, 0.1, 1.0]')
parser.add_argument('--early_stopping', default=[3, 5, 10], help='Default is [3, 5, 10]')
parser.add_argument('--epochs', type=int, default=[200], help='Default is 200')
parser.add_argument('--seed', type=int, default=[0, 1, 2, 3, 4], help='Default is [0 ,1 ,2 ,3 ,4 ]')
parser.add_argument('--leaf-level', type=str, default='genome_id', help='taxonomical level down to which the tree will be built')
parser.add_argument('--model-to-run', type=str, default='both', help='both, dendronet or logistic')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--force-train', type=str, default='y', help='Decide if you want the model to recompute for combination that were trained already')
parser.add_argument('--gpu_mode', type=str, default='single', help='Using a single or multiple GPUs')
args = parser.parse_args()
if __name__ == "__main__":

    # making some adjustments based on results from previous tuning
    dpf_list = args.dpfs
    lr_list = args.lrs
    l1_list = args.l1s
    e_stop_list = args.early_stopping
    epoch_list = args.epochs
    if args.gpu_mode == 'single':
        exp_file = 'experiment3.py'
    elif args.gpu_mode == 'multiple':
        exp_file = 'experiment_multi_gpu.py'

    if args.model_to_run == 'both' or args.model_to_run == 'dendronet':
        print("DendroNet")
        for dpf in dpf_list:
            for lr in lr_list:
                for epoch in epoch_list:
                    for l1 in l1_list:
                        for e_stop in e_stop_list:

                            dir_name = args.antibiotic + '_' + args.group + '_dendronet_' + str(dpf) + '_' + str(lr) + '_' \
                                       + str(l1) + '_' + str(e_stop) + '_' + args.leaf_level

                            output_path = os.path.join('data_files', 'patric_tuning', dir_name, 'output.json')
                            print(dir_name)
                            if not os.path.isdir(os.path.join('data_files', 'patric_tuning', dir_name)) or args.force_train == 'y':
                                command = 'python ' + exp_file + ' --epochs ' + str(epoch) + ' --dpf ' + str(dpf) \
                                            + ' --early-stopping ' + str(e_stop) + ' --lr ' + str(lr) + ' --output-path ' + output_path \
                                            + ' --l1 ' + str(l1) + ' --lineage-path ' + str(args.genome_lineage) \
                                            + ' --leaf-level ' + str(args.leaf_level) \
                                            + ' --batch-size ' + str(args.batch_size) \
                                            + ' --label-file ' + os.path.join('data_files', 'subproblems',
                                                                              args.group + '_' + args.antibiotic,
                                                                              args.group + '_' + args.antibiotic + '_samples.csv')
                                          # + ' --seed ' + str(args.seed)
                                os.system(command)

        df, results = build_tab(antibiotic=args.antibiotic, group=args.group, model='dendronet', leaf_level=args.leaf_level)
        """
        for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
            if args.group in directory and args.antibiotic in directory and 'dendronet' in directory and args.leaf_level in directory:
                os.system('rm -r ' + os.path.join('data_files', 'patric_tuning', directory))
        """

    if args.model_to_run == 'both' or args.model_to_run == 'logistic':
        print("Logistic")
        for lr in lr_list:
            for epoch in epoch_list:
                for e_stop in e_stop_list:
                    dir_name = args.antibiotic + '_' + args.group + '_logistic_' \
                               + str(lr) + '_' + str(e_stop)

                    output_path = os.path.join('data_files', 'patric_tuning', dir_name, 'output.json')
                    print(dir_name)
                    if not os.path.isdir(os.path.join('data_files', 'patric_tuning', dir_name)) or args.force_train == 'y':
                        command = 'python log_experiment.py --epochs ' + str(epoch)  \
                                  + ' --early-stopping ' + str(e_stop) + ' --lr ' + str(lr)  \
                                  + ' --output-path ' + output_path \
                                  + ' --batch-size ' + str(args.batch_size) \
                                  + ' --label-file ' + os.path.join('data_files', 'subproblems',
                                                                    args.group + '_' + args.antibiotic,
                                                                    args.group + '_' + args.antibiotic + '_samples.csv')
                        # + ' --seed ' + str(args.seed)
                        os.system(command)

        df, results = build_tab(antibiotic=args.antibiotic, group=args.group, model='logistic', leaf_level='none')
        """
        for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
            if args.group in directory and args.antibiotic in directory and 'logistic' in directory and args.leaf_level in directory:
                os.system('rm -r ' + os.path.join('data_files', 'patric_tuning', directory))
        """












