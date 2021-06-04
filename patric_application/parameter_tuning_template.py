import os

# making some adjustments based on results from previous tuning

dpf_list = [0.001, 0.01, 0.1, 1.0]
lr_list = [0.01, 0.001, 0.0001]
l1_list = [0.0, 0.01, 0.1, 1.0]
epoch_list = [100]

tree_path = ''
feature_csv_path = ''

if __name__ == "__main__":
    for dpf in dpf_list:
        for lr in lr_list:
            for epoch in epoch_list:
                for l1 in l1_list:
                    output_dir = 'patric_tuning_' + feature_csv_path.split('/')[-1] + '_' + str(dpf) + '_' + str(lr) + '_' \
                                 + str(l1)
                    command = 'python3 experiment.py --epochs ' + str(epoch) + ' --dpf ' + str(dpf) \
                              + ' --lr ' + str(lr) + ' --output-dir ' + output_dir \
                              + ' --l1 ' + str(l1) #+ ARGS FOR DATA PATH
                    os.system(command)
