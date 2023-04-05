import numpy as np
from functools import cmp_to_key

def doc_comparator(doc1, doc2):
    if doc1[1] < doc2[1]:
        return 1
    elif doc1[1] == doc2[1]:
        return int(doc1[0] > doc2[0])
    else:
        return -1
    
def _get_first_k_documents(y_pred, y_true, k):
    assert y_pred.shape[0] == y_true.shape[0]

    first_k_docs = sorted(zip(y_true, y_pred), key=cmp_to_key(doc_comparator))
    first_k_docs = np.array(first_k_docs)[:k, 0]

    return first_k_docs

### NDCG EVALUATION ###

def cumulative_gain(relevances):
    return np.sum((2 ** relevances - 1) / np.log2(np.arange(relevances.shape[0]) + 2))

def ndcg(y_pred, y_true, k):
    k = min(k, y_pred.shape[0])
    first_k_docs = _get_first_k_documents(y_pred, y_true, k)

    top_k_idxs = np.argsort(y_true)[::-1][:k]
    top_k_docs = y_true[top_k_idxs]

    dcg = cumulative_gain(first_k_docs)
    idcg = cumulative_gain(top_k_docs)

    return dcg / idcg if idcg > 0 else 1.

def ndcg_score(y_pred, y_true, query_indices, k=10):
    sum_ndcg = 0
    queries = np.unique(query_indices)

    for query in queries:
        idxs = query_indices == query
        sum_ndcg += ndcg(y_pred[idxs], y_true[idxs], k)

    return sum_ndcg / float(queries.shape[0])

### MRR EVALUATION ###

def mrr(y_pred, y_true, min_relevant_rank, k):
    k = min(k, y_pred.shape[0])
    first_k_docs = _get_first_k_documents(y_pred, y_true, k)

    for idx, rel in enumerate(first_k_docs):
        if rel >= min_relevant_rank:
            return (1.0 / (idx + 1.0))
    
    return 0.0

def mrr_score(y_pred, y_true, query_indices, min_relevant_rank=2):
    queries = np.unique(query_indices)
    sum_mrr = 0

    for query in queries:
        idxs = query_indices == query
        k = len(y_true[idxs])
        sum_mrr += mrr(y_pred[idxs], y_true[idxs], min_relevant_rank, k)

    return sum_mrr / float(queries.shape[0])

### PRECISION EVALUATION ###

def precision_at_k(y_pred, y_true, min_relevant_rank, k):
    k = min(k, y_pred.shape[0])
    first_k_docs = _get_first_k_documents(y_pred, y_true, k)
    return (first_k_docs >= min_relevant_rank).sum() / k

def precision_score(y_pred, y_true, query_indices, k=10, min_relevant_rank=2):
    queries = np.unique(query_indices)
    sum_precision = 0

    for query in queries:
        idxs = query_indices == query
        value = precision_at_k(y_pred[idxs], y_true[idxs], min_relevant_rank, k)
        sum_precision += value

    return sum_precision / float(queries.shape[0])

def map_score(y_pred, y_true, query_indices, min_relevant_rank=2):
    queries = np.unique(query_indices)
    result = 0

    for query in queries:
        idxs = query_indices == query
        max_k = np.count_nonzero(idxs)

        current_sum = 0
        correct_predictions = 0

        for k in range(1, max_k + 1):
            # Calculate precision at current value of k
            p_at_k = precision_at_k(y_pred[idxs], y_true[idxs], min_relevant_rank, k)
            # If document at k is relevant
            if int(y_true[idxs][k - 1]) >= min_relevant_rank:
                # Increase the running sum by the value of precision at current k
                current_sum += p_at_k
                correct_predictions += 1

        # In division-by-zero scenario divide by 1 which gives a result of 0 either way
        correct_predictions = max(correct_predictions, 1)
        result += (current_sum / correct_predictions)

    # Calculate average over all queries
    return result / len(queries)

### RECALL EVALUATION ###

def recall_at_k(y_pred, y_true, min_relevant_rank, k):
    k = min(k, y_pred.shape[0])
    first_k_docs = _get_first_k_documents(y_pred, y_true, k)
    # Return 1 if no relevant documents exist so that the metric is equal to 0
    relevant = max((y_true >= min_relevant_rank).sum(), 1)
    relevant_retrieved = (first_k_docs >= min_relevant_rank).sum()

    return relevant_retrieved / relevant

def recall_score(y_pred, y_true, query_indices, k=10, min_relevant_rank=2):
    queries = np.unique(query_indices)
    sum_recall = 0

    for query in queries:
        idxs = query_indices == query
        value = recall_at_k(y_pred[idxs], y_true[idxs], min_relevant_rank, k)
        sum_recall += value

    return sum_recall / float(queries.shape[0])

### RMSE EVALUATION ###

def rmse(y_pred, y_true, max_relevance, k):
    k = min(k, y_pred.shape[0])
    first_k_docs = sorted(zip(y_true, y_pred), key=cmp_to_key(doc_comparator))[:k]
    
    error = 0
    for result in first_k_docs:
        error += (result[0] - max_relevance * result[1]) ** 2
    
    return error ** (1 / 2)

def rmse_score(y_pred, y_true, query_indices, k=10, max_relevance=2.0):
    queries = np.unique(query_indices)
    mse = 0

    for query in queries:
        idxs = query_indices == query
        value = rmse(y_pred[idxs], y_true[idxs], max_relevance, k)
        mse += value

    return mse / float(queries.shape[0])