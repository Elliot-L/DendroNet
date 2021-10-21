import os
import json

if __name__ == '__main__':
    for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
        file = os.path.join('data_files', 'patric_tuning', directory, 'output.json')
        with open(file, 'w') as f:
            dic = json.load(f)
        if len(dic) > 3:
            d = os.path.join('data_files', 'patric_tuning', directory)
            os.system('rm ' + d)

