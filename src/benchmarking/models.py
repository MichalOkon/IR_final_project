from collections import Counter
from metrics import ndcg_score, mrr_score, precision_score, recall_score, rmse_score
import lightgbm as lgb


class Data:
    def __init__(self, train, test, validation):
        self.X_train = train[0]
        self.y_train = train[1]
        self.queries_train = train[2]

        self.X_validation = validation[0]
        self.y_validation = validation[1]
        self.queries_validation = validation[2]

        self.X_test = test[0]
        self.y_test = test[1]
        self.queries_test = test[2]
        self.group_test = list(Counter(self.queries_test).values())

        group_train = list(Counter(self.queries_train).values())
        group_validation = list(Counter(self.queries_validation).values())
        self.train_pool = lgb.Dataset(self.X_train, self.y_train, group=group_train)
        self.validation_pool = lgb.Dataset(self.X_validation, self.y_validation, group=group_validation)


class Ranker:
    def evaluate(self, data):
        predictions = self.model.predict(data.X_test, group=data.group_test)
        # TODO: Update to include all of our metrics
        return {
            "ndcg": ndcg_score(data.y_test, predictions, len(data.y_test)),
            "mrr": mrr_score(data.y_test, predictions),
            "precision_at_1": precision_score(data.y_test, predictions, 1),
            "precision_at_5": precision_score(data.y_test, predictions, 5),
            "precision_at_10": precision_score(data.y_test, predictions, 10), 
            "recall_at_1": recall_score(data.y_test, predictions, 1),
            "recall_at_5": recall_score(data.y_test, predictions, 5),
            "recall_at_10": recall_score(data.y_test, predictions, 10),
            "rmse": rmse_score(data.y_test, predictions)
        }

    def fit(self, train):
        raise Exception('Call of interface function')

    def staged_predict(self, data, eval_period):
        raise Exception('Call of interface function')


class LightGBMRanker(Ranker):
    def __init__(self, params):
        self.params = params

        self.iterations = self.params['iterations']
        del self.params['iterations']

    def fit(self, data):
        self.model = lgb.train(
            params=self.params,
            train_set=data.train_pool,
            valid_sets=data.validation_pool,
            num_boost_round=self.iterations
        )
        self.n_iter_ = self.model.current_iteration()

    def staged_predict(self, data, eval_period):
        staged_predictions = []
        for i in range(0, self.iterations, eval_period):
            prediction = self.model.predict(data.X_test, i + 1, group=data.group_test)
            staged_predictions.append(prediction)

        return staged_predictions