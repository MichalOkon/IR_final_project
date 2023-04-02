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


def argmin(fn, space):
    best_score = np.NINF
    best_params = {}

    for params in ParameterGrid(space):
        try:
            score = fn(params)
        except Exception as e:
            print('Exception during training: ' + repr(e))
            continue

        if score > best_score:
            best_score = score
            best_params = params

    return {'best_score': best_score, 'best_params': best_params}


def _params_to_str(params):
    return ''.join(map(lambda t: '{}[{}]'.format(t[0], str(t[1])), params.items()))


def eval_params(ranker_name, RankerType, data, static_params, param_space, log_file, out_file):
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump({}, f)

    def objective(params):
        ranker_params = deepcopy(static_params)
        ranker_params.update(params)
        print('Fit with params: ' + _params_to_str(ranker_params))

        with open(log_file, 'r') as f:
            log = json.load(f)
            if ranker_name not in log:
                log[ranker_name] = {}
            params_str = _params_to_str(params)
            if params_str in log[ranker_name]:
                print('Return result from cache')
                return max(log[ranker_name][params_str]['ndcg'])

        ranker = RankerType(ranker_params)

        start = datetime.datetime.now()

        ranker.fit(data)

        train_time = datetime.datetime.now() - start
        train_time = train_time.total_seconds()

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

    best_params = argmin(fn=objective, space=param_space)

    print('Best params:' + str(best_params))

    with open(out_file, 'w') as f:
        json.dump(best_params, f, indent=4, sort_keys=True)

    return best_params


def print_versions():
    import lightgbm
    print('LightGBM: ' + lightgbm.__version__)


# TODO: Adapt the arguments
if __name__ == "__main__":
    rankers = {
        'lgb-rmse': [LightGBMRanker, 'regression'],
        'lgb-pairwise': [LightGBMRanker, 'lambdarank'],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--learner', choices=rankers.keys(), required=True)
    # For example "MQ2008/Fold1"
    parser.add_argument('--fold', required=True)
    # parser.add_argument('-s', '--param-space', required=True)
    # parser.add_argument('-o', '--out-file', required=True)
    # parser.add_argument('-l', '--log-file', required=True)
    parser.add_argument('-n', '--iterations', type=int, default=10000)
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()

    print_versions()

    train_path = "/".join(["../../data/parsed", args.fold, "train.txt"])
    test_path = "/".join(["../../data/parsed", args.fold, "test.txt"])
    validation_path = "/".join(["../../data/parsed", args.fold, "vali.txt"])

    RankerType = rankers[args.learner][0]
    loss_function = rankers[args.learner][1]

    static_params = {
        'verbose': 0,
        'boosting_type': 'gbdt',
        'random_seed': RANDOM_SEED
    }

    if args.use_gpu:
        static_params['device'] = 'gpu'
        static_params['gpu_device_id'] = 0

    static_params['iterations'] = args.iterations

    # with open(args.param_space) as f:
    #     param_space = json.load(f)

    train = read_dataset(train_path)
    test = read_dataset(test_path)
    validation = read_dataset(validation_path)
    data = Data(train, test, validation)

    result = eval_params(args.learner, RankerType, data, static_params, param_space, args.log_file, args.out_file)
    print('NDCG best value: ' + str(result))