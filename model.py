import argparse
import os
from itertools import groupby
from hmmlearn import hmm
import gzip as gz
import pandas as pd
import numpy as np
import pickle

NUM_OF_STATS = 3
STATES = {"H": 0, "S": 1, "L": 2}
AMINO_ACIDS = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
               "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17,
               "Y": 18, "V": 19, "B": 20, "Z": 21, "X": 22, "J": 23, "O": 24, "U": 25, }
BAD_AMINO_ACIDS = "OUBZXJ"
EMISSIONS = pd.read_csv("emissions-Table 1.csv", index_col=0).T.to_numpy() / 100
TRANSITIONS = np.array([[12 / 13, 1 / 39, 2 / 39], [1 / 15, 4 / 5, 2 / 15], [0.4, 0.2, 0.4]])
PATH_TO_DATA = "/Users/alonbentzion/Downloads/parsed_sequences" # Local

def split_train_test(dir_path: str):
    train_samples, train_lengths = [], []
    test_samples, test_lengths, test_labels = [], [], []

    for file in os.listdir(dir_path):
        flag = np.random.binomial(1, 0.8, 1)[0]
        file = f"{dir_path}/{file}"
        with open(file, 'r') as f:
            content = f.readlines()
            seq, labels = content[0], content[1]
            numeric_seq, numeric_labels = [], []
            for s, l in zip(seq, labels):
                if s not in BAD_AMINO_ACIDS:
                    numeric_seq += [AMINO_ACIDS[s]]
                    numeric_labels += [STATES[l]]
            if flag:
                train_samples.append(np.array(numeric_seq).reshape(-1, 1))
                train_lengths.append(len(numeric_seq))
            else:
                test_samples.append(np.array(numeric_seq).reshape(-1, 1))
                test_lengths.append(len(numeric_seq))
                test_labels.append(numeric_labels)

    return train_samples, train_lengths, test_samples, test_lengths, test_labels


def train(train_samples, train_lengths, convergence_threshold: float = 0.01, num_iters: int = 80):
    # Create HMM model
    model = hmm.CategoricalHMM(n_components=NUM_OF_STATS,
                               algorithm='viterbi',
                               n_iter=num_iters,
                               tol=convergence_threshold,
                               verbose=True,
                               params="et",
                               init_params="s")

    # Config emissions
    model.emissionprob_ = EMISSIONS
    model.transmat_ = TRANSITIONS
    # Train th model
    model.fit(np.concatenate(train_samples).reshape((-1, 1)), train_lengths)

    return model


def main():
    # Get train and test samples
    train_samples, train_lengths, test_samples, test_lengths, test_labels = split_train_test()

    errors_rate = []
    for iter in range(1, 100, 2):
        # Create and train model
        model = train(train_samples, train_lengths, num_iters=iter)

        # Predict
        misclassification_error = 0
        for sample, length, label in zip(test_samples, test_lengths, test_labels):
            ll, hidden_states = model.decode(sample, length)
            misclassification_error += np.count_nonzero(hidden_states - label)
        # Save misclassification error
        misclassification_error_rate = misclassification_error / np.sum(test_lengths)
        errors_rate.append(misclassification_error_rate)

        # Create pickle file
        with open(f"models/model_{iter}", 'wb') as file:
            pickle.dump(model, file)



if __name__ == '__main__':
    main()
