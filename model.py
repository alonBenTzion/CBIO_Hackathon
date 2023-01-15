import argparse
import os
from itertools import groupby
from hmmlearn import hmm
import gzip as gz
import pandas as pd
import numpy as np

NUM_OF_STATS = 3
STATES = {"H": 0, "S": 1, "L": 2}
AMINO_ACIDS = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
               "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "O": 15, "S": 16, "U": 17, "T": 18, "W": 19,
               "Y": 20, "V": 21, "B": 22, "Z": 23, "X": 24, "J": 25}
BAD_AMINO_ACIDS = "OUBZXJ"
EMISSIONS = np.recfromcsv("emissions-Table 1.csv")


def split_train_test(dir_path: str):
    train_samples, train_lengths = [], []
    test_samples, test_lengths, test_labels = [], [], []

    for file in os.listdir(dir_path):
        flag = np.random.binomial(1, 0.8, 1)
        with open(file, 'r') as f:
            content = f.readlines()
            seq, labels = content[0], content[1]
            numeric_seq, numeric_labels = [], []
            for s, l in zip(seq, labels):
                if s not in BAD_AMINO_ACIDS:
                    numeric_seq += [AMINO_ACIDS[s]]
                    numeric_labels += STATES[l]
            if flag:
                train_samples.append(numeric_seq)
                train_lengths.append(len(numeric_seq))
            else:
                test_samples.append(numeric_seq)
                test_lengths.append(len(numeric_seq))
                test_labels.append(numeric_labels)

    return train_samples, train_lengths, test_samples, test_lengths, test_labels


def train(train_samples, train_lengths, convergence_threshold: float = 0.01, num_iters: int = 120):
    # Create HMM model
    model = hmm.CategoricalHMM(n_components=NUM_OF_STATS,
                               algorithm='viterbi',
                               n_iter=num_iters,
                               tol=convergence_threshold,
                               verbose=True,
                               params="e",
                               init_params="st")

    # Config emissions
    model.emissionprob_ = EMISSIONS
    # Train th model
    model.fit(np.concat(train_samples).reshape((-1, 1)), train_lengths)

    return model


def predict_fasta(model, fasta):
    states = []
    likelihoods = []
    lengths = []

    reader = fastaread_gz if is_gz_file(fasta) else fastaread
    for _, seq in reader(fasta):
        parsed_seq = np.array([[BASES_INDICES[c]] for c in seq])
        seq_len = np.array([len(seq)])
        ll, hidden_states = model.decode(parsed_seq, seq_len)
        hidden_states = np.where(hidden_states >= 4, 'N', 'I')
        states.append(hidden_states)
        likelihoods.append(ll)
        lengths.append(len(seq))

    states = [''.join(s.astype(str)) for s in states]

    return states, likelihoods, lengths


def main():
    # Get train and test samples
    train_samples, train_lengths, test_samples, test_lengths, test_labels = split_train_test("DIR_PATH")

    # Create and train model
    model = train(train_samples, train_lengths)

    # Predict
    misclassification_error = 0
    for sample, length, label in zip(test_samples, test_lengths, test_labels):
        ll, hidden_states = model.decode(sample, length)
        misclassification_error += np.count_nonzero(hidden_states - label)
    misclassification_error_rate = misclassification_error / np.sum(test_lengths)

if __name__ == '__main__':
    main()
