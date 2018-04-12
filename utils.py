from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas

def cross_valid(models, feats, labels):
    ns = 3
    k_fold = KFold(n_splits=ns)
    scores = []
    for clf in models:
        score = sum(cross_val_score(clf, feats, labels, cv=k_fold,
        	scoring='neg_mean_squared_error')) / ns
        scores.append(-1 * score)
    scores = np.sqrt(scores)
    return scores

def write_test_file(preds, test_idx):
    d = {'Id': test_idx, 'SalePrice': preds}
    df = pandas.DataFrame(data=d)
    df.to_csv('data/submit.csv')