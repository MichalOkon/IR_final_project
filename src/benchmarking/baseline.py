import argparse
import datetime
import json
import os
import numpy as np

from copy import deepcopy
from models import *
from utils import read_dataset
from sklearn.model_selection import ParameterGrid

RANDOM_SEED = 0

def run_grid_search(objective, param_space):
    """Assess the performance on every combination of parameters supplied from a JSON file.
    """
    best_score = np.NINF
    best_params = {}

    for params in ParameterGrid(param_space):
        try:
            score = objective(params)
        except Exception as e:
            print(f"Exception during training: {repr(e)}")
            continue

        if score > best_score:
            best_score = score
            best_params = params

    return {'best_score': best_score, 'best_params': best_params}


def _params_to_str(params):
    """Parse the parameters into a more human-friendly format.
    """
    return '--'.join(map(lambda t: '{}[{}]'.format(t[0], str(t[1])), params.items()))


def run_training(ranker_name, ranker_type, data, static_params, param_space, log_file, out_file):
    """Execute the complete evaluation sequence.
    """

    # Initialize the log file in case it does not exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump({}, f)

    def objective(params):
        """Execute the training sequence for a single model and store its hyper-parameters into the log file.
        """
        # Combine static hyper-parameters (number of iterations) with user-specified hyper-parameters
        ranker_params = deepcopy(static_params)
        ranker_params.update(params)
        print(f"Fit with params: {_params_to_str(ranker_params)}")

        # Add a new entry to the log file
        with open(log_file, 'r') as f:
            log = json.load(f)
            if ranker_name not in log:
                log[ranker_name] = {}
            params_str = _params_to_str(params)
            if params_str in log[ranker_name]:
                print('Return result from cache')
                return max(log[ranker_name][params_str]['ndcg'])

        # In our case this is actually always LightGBMRanker
        ranker = ranker_type(ranker_params)

        # Execute the training sequence
        print("Starting the training")
        start = datetime.datetime.now()
        ranker.fit(data)

        train_time = datetime.datetime.now() - start
        train_time = train_time.total_seconds()

        # Evaluate the final score
        # TODO: Update to take all of our metrics into consideration
        eval_log = ranker.eval_ndcg(data)

        log[ranker_name][params_str] = {
            'time': train_time,
            'ndcg': eval_log
        }

        dump = log_file + '.dmp'
        with open(dump, 'w') as f:
            json.dump(log, f, indent=4, sort_keys=True)
        os.rename(dump, log_file)

        return max(eval_log)

    best_params = run_grid_search(objective=objective, param_space=param_space)
    print('Best params:' + str(best_params))

    # Save the hyper-parameters of the best model found
    with open(out_file, 'w') as f:
        json.dump(best_params, f, indent=4, sort_keys=True)

    # TODO: Optimize for NDCG@5
    return best_params


# TODO: Adapt the arguments
if __name__ == "__main__":
    rankers = {
        'lgb-rmse': [LightGBMRanker, 'regression'],
        'lgb-pairwise': [LightGBMRanker, 'lambdarank'],
    }

    parser = argparse.ArgumentParser()
    # One of the models defined above
    parser.add_argument('--learner', choices=rankers.keys(), required=True)
    # For example "MQ2008/Fold1"
    parser.add_argument('--fold', required=True)

    parser.add_argument('-s', '--param-space', required=True)
    parser.add_argument('-o', '--out-file', required=True)
    parser.add_argument('-l', '--log-file', required=True)

    parser.add_argument('-n', '--iterations', type=int, default=10000)
    args = parser.parse_args()

    train_path = "/".join(["../../data/parsed", args.fold, "train.csv"])
    test_path = "/".join(["../../data/parsed", args.fold, "test.csv"])
    validation_path = "/".join(["../../data/parsed", args.fold, "vali.csv"])

    # DEBUG
    # train_path = "/".join(["data/parsed", args.fold, "train.csv"])
    # test_path = "/".join(["data/parsed", args.fold, "test.csv"])
    # validation_path = "/".join(["data/parsed", args.fold, "vali.csv"])

    ranker_type = rankers[args.learner][0]
    loss_function = rankers[args.learner][1]

    static_params = {
        'verbose': 0,
        'boosting_type': 'gbdt',
        'random_seed': RANDOM_SEED
    }

    static_params['iterations'] = args.iterations

    with open(args.param_space) as f:
        param_space = json.load(f)

    train = read_dataset(train_path)
    test = read_dataset(test_path)
    validation = read_dataset(validation_path)
    data = Data(train, test, validation)

    result = run_training(args.learner, ranker_type, data, static_params, param_space, args.log_file, args.out_file)
    print(f"NDCG best value: {str(result)}")