import numpy as np
from functools import cmp_to_key

def doc_comparator(doc1, doc2):
    if doc1[1] < doc2[1]:
        return 1
    elif doc1[1] == doc2[1]:
        return int(doc1[0] > doc2[0])
    else:
        return -1

def cumulative_gain(relevances):
    return np.sum((2 ** relevances - 1) / np.log2(np.arange(relevances.shape[0]) + 2))


def ndcg(y_pred, y_true, top):
    assert y_pred.shape[0] == y_true.shape[0]
    top = min(top, y_pred.shape[0])

    first_k_docs = sorted(zip(y_true, y_pred), key=cmp_to_key(doc_comparator))
    first_k_docs = np.array(first_k_docs)[:top,0]

    top_k_idxs = np.argsort(y_true)[::-1][:top]
    top_k_docs = y_true[top_k_idxs]

    dcg = cumulative_gain(first_k_docs)
    idcg = cumulative_gain(top_k_docs)

    return dcg / idcg if idcg > 0 else 1.


def mean_ndcg(y_pred, y_true, query_idxs, top=10):
    sum_ndcg = 0
    queries = np.unique(query_idxs)

    for query in queries:
        idxs = query_idxs == query
        value = ndcg(y_pred[idxs], y_true[idxs], top)
        sum_ndcg += value

    return sum_ndcg / float(queries.shape[0])