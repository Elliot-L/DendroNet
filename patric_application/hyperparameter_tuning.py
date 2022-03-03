import os
import argparse
from build_data_tab import build_tab
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--antibiotic', type=str, default='erythromycin', metavar='ANT')
parser.add_argument('--group', type=str, default='Firmicutes', metavar='GR')
parser.add_argument('--threshold', type=str, default='0.0', help='Threshold for feature selection')
parser.add_argument('--genome-lineage', type=str, default=os.path.join('data_files', 'genome_lineage.csv'))
parser.add_argument('--dpfs', type=float, nargs='+', default=[0.001, 0.01, 0.1], help='Default is [0.001, 0.01, 0.1, 1.0]')
parser.add_argument('--lrs', type=float, nargs='+', default=[0.01, 0.001, 0.0001], help='Default is [0.01, 0.001, 0.0001]')
parser.add_argument('--l1s', type=float, nargs='+', default=[0.0, 0.01, 0.1, 1.0], help='Default is [0.0, 0.01, 0.1, 1.0]')
parser.add_argument('--early_stopping', nargs='+', default=[5], help='Default is [3, 5, 10]')
#parser.add_argument('--dpfs', type=float, nargs='+', default=[0.001], help='Default is [0.001, 0.01, 0.1, 1.0]')
#parser.add_argument('--lrs', type=float, nargs='+', default=[0.01], help='Default is [0.01, 0.001, 0.0001]')
#parser.add_argument('--l1s', type=float, nargs='+', default=[0.0], help='Default is [0.0, 0.01, 0.1, 1.0]')
#parser.add_argument('--early_stopping', nargs='+', default=[3], help='Default is [3, 5, 10]')
parser.add_argument('--epochs', type=int, nargs='+', default=[1000], help='Default is 1000')
parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='Default is [0 ,1 ,2 ,3 ,4 ]')
parser.add_argument('--leaf-level', type=str, default='genomeID', help='taxonomical level down to which the tree will be built')
parser.add_argument('--model-to-run', type=str, default='both', help='both, dendronet or logistic')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--gpu-to-use', type=int, nargs='+', default=[0])
parser.add_argument('--force-train_log', type=str, default='n', help='Decide if you want the model to recompute for'
                                                                     ' combination that were trained already on the '
                                                                     'logistic regression')
parser.add_argument('--force-train-dendronet', type=str, default='y', help='Decide if you want the model to recompute '
                                                                           'for combination that were trained already'
                                                                           'on dendronet')
#parser.add_argument('--gpu-mode', type=str, default='single', help='Using a single or multiple GPUs')
args = parser.parse_args()

if __name__ == "__main__":

    dpf_list = args.dpfs
    lr_list = args.lrs
    l1_list = args.l1s
    e_stop_list = args.early_stopping
    epoch_list = args.epochs
    """
    if args.gpu_mode == 'single':
        exp_file = 'experiment.py'
    elif args.gpu_mode == 'multiple':
        exp_file = 'experiment_multi_gpu.py'
    

    if torch.cuda.is_available():
        Cuda_str = 'CUDA_VISIBLE_DEVICES='
        for i, gpu_id in enumerate(args.gpu_to_use):
            Cuda_str + str(gpu_id)
            if i == (len(args.gpu_to_use) - 1):
                Cuda_str + ' '
            else:
                Cuda_str + ','
    else:
        Cuda_str = ''
    """

    exp_file = 'dendronet_experiment.py'

    seeds_str = ''
    for s in args.seeds:
        seeds_str += ' '
        seeds_str += str(s)

    if args.model_to_run == 'both' or args.model_to_run == 'dendronet':
        print("Dendronet")
        for dpf in dpf_list:
            for lr in lr_list:
                for l1 in l1_list:
                    for e_stop in e_stop_list:
                        for epoch in epoch_list:

                            dir_name = args.group + '_' + args.antibiotic + '_' + args.threshold \
                                       + '_dendronet_' + str(dpf) + '_' + str(lr) \
                                       + '_' + str(l1) + '_' + str(e_stop) \
                                       + '_' + leaf_level

                            output_path = os.path.join('data_files', 'patric_tuning', dir_name, 'output.json')
                            print(dir_name)
                            if not os.path.isdir(os.path.join('data_files', 'patric_tuning', dir_name)) or args.force_train_dendronet == 'y':
                                command = 'python ' + exp_file \
                                          + ' --epochs ' + str(epoch) \
                                          + ' --dpf ' + str(dpf) \
                                          + ' --l1 ' + str(l1) \
                                          + ' --lr ' + str(lr) \
                                          + ' --early-stopping ' + str(e_stop) \
                                          + ' --output-path ' + output_path \
                                          + ' --lineage-path ' + args.genome_lineage \
                                          + ' --leaf-level ' + leaf_level \
                                          + ' --batch-size ' + str(args.batch_size) \
                                          + ' --group ' + args.group \
                                          + ' --antibiotic ' + args.antibiotic \
                                          + ' --threshold ' + args.threshold \
                                          + ' --seeds' + seeds_str
                                os.system(command)

        df, results = build_tab(antibiotic=args.antibiotic, group=args.group, model='dendronet',
                                leaf_level=leaf_level, threshold=args.threshold, seeds=args.seeds)
        """ 
        This code was planned to be used as a way to avoid accumulation of data that has been used
        for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
            if args.group in directory and args.antibiotic in directory and 'dendronet' in directory and args.leaf_level in directory:
                os.system('rm -r ' + os.path.join('data_files', 'patric_tuning', directory))
        """

    if args.model_to_run == 'both' or args.model_to_run == 'logistic':
        print("Logistic")
        for lr in lr_list:
            for l1 in l1_list:
                for epoch in epoch_list:
                    for e_stop in e_stop_list:
                        dir_name = args.group + '_' + args.antibiotic + '_' + str(args.threshold) + '_logistic_' \
                                   + str(lr) + '_' + str(l1) + '_' + str(e_stop)

                        output_path = os.path.join('data_files', 'patric_tuning', dir_name, 'output.json')
                        print(dir_name)
                        if not os.path.isdir(os.path.join('data_files', 'patric_tuning', dir_name)) or args.force_train_log == 'y':
                            command = 'python logistic_experiment.py ' \
                                      + '--epochs ' + str(epoch)  \
                                      + ' --early-stopping ' + str(e_stop) \
                                      + ' --lr ' + str(lr)  \
                                      + ' --l1 ' + str(l1) \
                                      + ' --output-path ' + output_path \
                                      + ' --batch-size ' + str(args.batch_size) \
                                      + ' --group ' + args.group \
                                      + ' --antibiotic ' + args.antibiotic \
                                      + ' --threshold ' + args.threshold \
                                      + ' --seeds' + seeds_str
                            os.system(command)

        df, results = build_tab(antibiotic=args.antibiotic, group=args.group, model='logistic',
                                leaf_level='none', threshold=args.threshold, seeds=args.seeds)
        """
        for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
            if args.group in directory and args.antibiotic in directory and 'logistic' in directory and args.leaf_level in directory:
                os.system('rm -r ' + os.path.join('data_files', 'patric_tuning', directory))
        
        """














