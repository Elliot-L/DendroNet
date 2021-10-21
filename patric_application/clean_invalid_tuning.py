import os
import json

if __name__ == '__main__':
    for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
        print(directory)
        file = os.path.join('data_files', 'patric_tuning', directory, 'output.json')
        if os.path.isfile(file):
            with open(file) as f:
                dic = json.load(f)
            if len(dic) > 3:
                d = os.path.join('data_files', 'patric_tuning', directory)
                os.system('rm -r ' + d)
        else:
            d = os.path.join('data_files', 'patric_tuning', directory)
            os.system('rm ' + d)

