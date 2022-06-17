import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

NUC_VOCAB = ['A', 'C', 'G', 'T', 'N']  # N for unknown nucleotides
LEN_NUC_VOCAB = len(NUC_VOCAB)
NUC_TO_IDX = {c: i for i, c in enumerate(NUC_VOCAB)}


def parse_fasta_file(filepath):
    '''
    This function parse fasta file and returns DNA sequences and their targets
    '''
    seqs, labels = [], []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.rstrip()
            if len(line) == 0:
                continue
            if line.startswith('>'):
                labels.append(float(line.split()[-1].split(':')[-1]))
            else:
                seqs.append(line.rstrip().upper())
    return seqs, labels


class DNATaskDataLoader:

    def __init__(self, all_seq, all_labels, batch_size, num_workers=4, shuffle=True):
        self.all_seq = np.array(all_seq)
        self.all_labels = np.array(all_labels)
        self.size = len(self.all_seq)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def __iter__(self):

        if self.shuffle:
            shuffled_idx = np.random.permutation(self.size)
            all_seq = self.all_seq[shuffled_idx]
            all_labels = self.all_labels[shuffled_idx]
        else:
            all_seq = self.all_seq
            all_labels = self.all_labels

        batches = [[all_seq[i: i + self.batch_size], all_labels[i: i + self.batch_size]] for i in
                   range(0, self.size, self.batch_size)]

        dataset = DNATaskDataset(batches)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                collate_fn=lambda x: x[0])

        for b in dataloader:
            yield b

        del batches, dataset, dataloader


class DNATaskDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_seq, batch_label = self.data[idx]
        # length of DNA sequence in a batch
        batch_len = torch.as_tensor(np.array([len(seq) for seq in batch_seq], dtype=np.int64))
        # convert DNA sequences to one-hot encodings, and pad (with zero vectors) all DNA sequences
        # in a batch to uniform length
        encoded_seq = encoding_rna_seq(batch_seq)
        # l2 loss target
        label = torch.as_tensor(np.array(batch_label, dtype=np.float32)).unsqueeze(1)
        return encoded_seq, label, batch_len


def encoding_rna_seq(batch_seq):
    encoded_seq = []
    max_len = max([len(seq) for seq in batch_seq])
    for seq in batch_seq:
        idx_seq = np.array([NUC_TO_IDX[c] for c in seq])
        one_hot_seq = np.eye(LEN_NUC_VOCAB, dtype=np.float32)[idx_seq]
        padding = np.zeros((max_len - len(seq), LEN_NUC_VOCAB), dtype=np.float32)
        encoded_seq.append(np.concatenate([one_hot_seq, padding], axis=0))
    return torch.as_tensor(np.stack(encoded_seq, axis=0))
