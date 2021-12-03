import os
import argparse




if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--change', type=str)
    parser.add_argument('--directory', type=str)
    args = parser.parse_args()
    if args.change == 'genome':
        for d in os.listdir(args.directory):
            name_list = d.split('_')
            if len(name_list) >= 9 and name_list[7] == 'genome' and name_list[8] == 'id':
                new_name_list = name_list.copy()
                new_name_list.pop(8)
                old_name = name_list[0]
                new_name = new_name_list[0]
                for i in range(1, len(name_list)):
                    old_name += '_' + name_list[i]
                for i in range(1, len(new_name_list)):
                    new_name += '_' + new_name_list[i]
                print(old_name)
                print(new_name)
                os.system('mv ' + old_name + ' ' + new_name)
    elif args.change == 'Bacteria':
        for d in os.listdir(args.directory):
            name_list = d.split('_')
            new_name_list = name_list.copy()
            if name_list[0] == 'All':
                new_name_list[0] == 'Bacteria'
            elif name_list[1] == 'All':
                new_name_list[1] = 'Bacteria'
            old_name = name_list[0]
            new_name = new_name_list[0]
            for i in range(1, len(name_list)):
                old_name += '_' + name_list[i]
                new_name += '_' + new_name_list[i]
            print(old_name)
            print(new_name)
            #os.system('mv ')
    elif args.change == 'threshold':
        for d in os.listdir(args.directory):
            name_list = d.split('_')
            new_name_list = name_list.copy()
            if len(name_list) == 8:
                new_name_list.append()
    elif args.change == 'group_antibiotic_order':
        pass
    elif args.change == 'capitalization':
        for d in os.listdir(args.directory):
            name_list = d.split('_')
            new_name_list = name_list.copy()
            if name_list[1] == 'firmicutes':
                new_name_list[1] = 'Firmicutes'
            old_name = name_list[0]
            new_name = new_name_list[0]
            for i in range(1, len(name_list)):
                old_name += '_' + name_list[i]
                new_name += '_' + new_name_list[i]
            print(old_name)
            print(new_name)
            #os.system('mv ')

