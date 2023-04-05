import numpy as np

from metrics import ndcg_score, mrr_score, precision_score, recall_score, rmse_score, map_score

### TEST NDCG ###

def test_ndcg_score():
    y_pred, y_true, queries = get_example_data()
    # For query 1: DCG = 4.93, IDCG = 5.39, NDCG = 0.915
    # For query 2: DCG = 2.39, IDCG = 3.63, NDCG = 0.658
    # For query 3: DCG = 1.63, IDCG = 1.63, NDCG = 1.000
    # Expected 0.857
    ndcg_at_4 = ndcg_score(y_pred, y_true, queries, k=4)

    np.testing.assert_almost_equal(ndcg_at_4, 0.857, decimal=3)

### TEST MRR ###

def test_mrr_score():
    y_pred, y_true, queries = get_example_data()
    # For query 1: 1 / 1 = 1.0
    # For query 2: 1 / 2 = 0.5
    # For query 3: 0

    mrr = mrr_score(y_pred, y_true, queries, min_relevant_rank=2)

    np.testing.assert_almost_equal(mrr, 0.500, decimal=3)

def test_mrr_simple_2():
    y_pred = np.array([0.5, 0.2])
    y_true = np.array([0.0, 1.0])
    queries = np.array([1, 1])
    result = mrr_score(y_pred, y_true, queries, min_relevant_rank=1)

    assert result == 0.5
### TEST PRECISION ###

def test_precision_score():
    y_pred, y_true, queries = get_example_data()
    # For query 1 first item is relevant
    # For query 2 first item is not relevant
    # For query 3 first item is not relevant
    # Expected 0.333
    precision_at_1 = precision_score(y_pred, y_true, queries, k=1, min_relevant_rank=2)
    # For query 1 two items at k = 4 are relevant -- 2 / 4 = 0.5
    # For query 2 one item at k = 4 is relevant -- 1 / 4 = 0.25
    # For query 3 zero items at k = 4 are relevant -- 0 / 4 = 0
    # Expected 0.375
    precision_at_4 = precision_score(y_pred, y_true, queries, k=4, min_relevant_rank=2)

    np.testing.assert_almost_equal(precision_at_1, 0.333, decimal=3)
    np.testing.assert_almost_equal(precision_at_4, 0.250, decimal=3)

### TEST RECALL ###

def test_recall_score():
    y_pred, y_true, queries = get_example_data()
    # For query 1 two items are relevant and we find one at k = 1 -- 1 / 2 = 0.5
    # For query 2 one item is relevant and we find zero at k = 1 -- 0
    # For query 3 zero items are relevant and we find zero at k = 1 -- 0
    # Expected 0.5
    recall_at_1 = recall_score(y_pred, y_true, queries, k=1, min_relevant_rank=2)
    # For query 1 two items are relevant and we find two at k = 4 -- 2 / 2 = 1
    # For query 2 one item is relevant and we find one at k = 4 -- 1 / 1 = 1
    # For query 3 zero items are relevant and we find zero at k = 4 -- 0
    # Expected 0.375
    recall_at_4 = recall_score(y_pred, y_true, queries, k=4, min_relevant_rank=2)

    np.testing.assert_almost_equal(recall_at_1, 0.166, decimal=3)
    np.testing.assert_almost_equal(recall_at_4, 0.666, decimal=3)

### TEST RMSE ###

def test_rmse_score():
    y_pred, y_true, queries = get_example_data()
    # For query 1: (((2 - 2 * 0.9)^2 + (0 - 2 * 0.85)^2 + (2 - 2 * 0.71)^2 + (1 - 2 * 0.63)^2)/4)^(1/2) = 0.913
    # For query 2: (((0 - 2 * 0.87)^2 + (2 - 2 * 0.76)^2 + (1 - 2 * 0.64)^2 + (0 - 2 * 0.26)^2)/4)^(1/2) = 0.9495
    # For query 3: (((1 - 2 * 0.7)^2 + (1 - 2 * 0.65)^2 + (0 - 2 * 0.32)^2 + (0 - 2 * 0.1)^2)/4)^(1/2) = 0.418
    # Expected: 0.76
    rmse = rmse_score(y_pred, y_true, queries, max_relevance=2)

    np.testing.assert_almost_equal(rmse, 0.76, decimal=3)

### TEST MAP ###

def test_map_score():
    y_pred, y_true, queries = get_example_data()
    # For query 1: [(1 / 1) + (0 / 2) + (2 / 3) + (0 / 4)] / 2 = 0.833
    # For query 2: [(0 / 1) + (1 / 2) + (0 / 3) + (0 / 4)] / 1 = 0.500
    # For query 3: 0
    # Expected: 0.444
    map = map_score(y_pred, y_true, queries, min_relevant_rank=2)

    np.testing.assert_almost_equal(map, 0.444, decimal=3)

### UTILITIES ###

def get_example_data():
    predictions = np.array([0.9, 0.85, 0.71, 0.63,
                            0.87, 0.76, 0.64, 0.26,
                            0.7, 0.65, 0.32, 0.1])
    
    ground_truth = np.array([2.0, 0.0, 2.0, 1.0,
                             0.0, 2.0, 1.0, 0.0,
                             1.0, 1.0, 0.0, 0.0])
    
    queries = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    return predictions, ground_truth, queries


if __name__ == "__main__":
    print("Testing NDCG...")
    test_ndcg_score()
    print("Testing MRR...")
    test_mrr_score()
    print("Testing Precision...")
    test_precision_score()
    print("Testing Recall...")
    test_recall_score()
    print("Testing RMSE...")
    test_rmse_score()
    print("Testing MAP...")
    test_map_score()