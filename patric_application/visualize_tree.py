import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str)
    parser.add_argument('--antibiotic', type=str)
    parser.add_argument('--leaf-level', type=str)

    args = parser.parse_args()

    matrix_file_name =
