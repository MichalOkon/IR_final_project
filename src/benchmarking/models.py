from collections import Counter
from metrics import mean_ndcg
import lightgbm as lgb


class Data:
    def __init__(self, train, test, validation):
        self.X_train = train[0]
        self.y_train = train[1]
        self.queries_train = train[2]

        print(validation)
        self.X_validation = validation[0]
        self.y_validation = validation[1]
        self.queries_validation = validation[2]

        print(test)
        self.X_test = test[0]
        self.y_test = test[1]
        self.queries_test = test[2]
        self.group_test = Counter(self.queries_test).values()

        group_train = Counter(self.queries_train).values()
        group_validation = Counter(self.queries_validation).values()
        
        self.train_pool = lgb.Dataset(self.X_train, self.y_train, group=group_train)
        self.validation_pool = lgb.Dataset(self.X_validation, self.y_validation, group=group_validation)


class Ranker:
    def eval_ndcg(self, data, eval_period=10):
        staged_predictions = self.staged_predict(data, eval_period)

        eval_log = []
        for y_pred in staged_predictions:
            # Here go the evaluation metrics
            value = mean_ndcg(y_pred, data.y_test, data.queries_test)
            eval_log.append(value)

        return eval_log

    def fit(self, train):
        raise Exception('Call of interface function')

    def staged_predict(self, data, eval_period):
        raise Exception('Call of interface function')


class LightGBMRanker(Ranker):
    def __init__(self, params):
        self.params = params

        self.params['num_leaves'] = 2 ** self.params['max_depth']
        del self.params['max_depth']

        self.iterations = self.params['iterations']
        del self.params['iterations']

    # Here feval parameter will include the custom evaluation function
    def fit(self, data):
        print(data.train_pool)
        self.model = lgb.train(
            params=self.params,
            train_set=data.train_pool,
            valid_sets=data.validation_pool,
            num_boost_round=self.iterations
        )

    def staged_predict(self, data, eval_period):
        staged_predictions = []
        for i in range(0, self.iterations, eval_period):
            prediction = self.model.predict(data.X_test, i + 1, group=data.group_test)
            staged_predictions.append(prediction)

        return staged_predictions