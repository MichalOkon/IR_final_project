import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

def process_libsvm_file(file_name):
    X, y, queries = load_svmlight_file(file_name, query_id=True)
    return X.todense(), y, queries


def dump_to_file(path, name, X, y, queries):
    all = np.hstack((y.reshape(-1, 1), queries.reshape(-1, 1), X))

    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)

    pd.DataFrame(all).sort_values(by=[1]).to_csv("\\".join([path, name]), sep=',', header=False, index=False)


def parse_dataset(src_path, dst_path, name):
    train_file = os.path.join(src_path, "train.txt")
    validation_file = os.path.join(src_path, "vali.txt")
    test_file = os.path.join(src_path, "test.txt")

    X, y, queries = process_libsvm_file(train_file)
    dump_to_file(dst_path, "train.csv", X, y, queries)

    X, y, queries = process_libsvm_file(validation_file)
    dump_to_file(dst_path, "vali.csv", X, y, queries)

    X, y, queries = process_libsvm_file(test_file)
    dump_to_file(dst_path, "test.csv", X, y, queries)

# Process all datasets to a format accepted in the rest of the framework
if __name__ == "__main__":
    datasets = ["MQ2008", "MSLR-WEB10K"]
    folds = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
    data_path = "..\\..\\data"

    for dataset in datasets:
        for fold in folds:
            src_path = "\\".join([data_path, dataset, fold])
            dst_path = "\\".join([data_path, "parsed", dataset, fold])
            parse_dataset(src_path, dst_path, dataset)