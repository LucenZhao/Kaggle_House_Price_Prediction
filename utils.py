from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas
import config

def cross_valid(models, feats, labels):
    ns = config.cv_folds
    k_fold = KFold(n_splits=ns)
    scores = []
    for i, clf in models:
        score = sum(cross_val_score(clf, feats, labels, cv=k_fold,
        	scoring='neg_mean_squared_error')) / ns
        scores.append(-1 * score)
    scores = np.sqrt(scores)
    return scores

def write_test_file(preds, test_idx, filename):
    d = {'Id': test_idx, 'SalePrice': preds}
    df = pandas.DataFrame(data=d)
    df.to_csv('data/'+filename)

def write_selection_results(f, name, methods, scores, names):
    f.write(name+'\n')
    for i in range(len(scores)):
        f.write(methods[i]+'\t\t'+str(scores[i])+'\n')
    f.write("BEST FEATURE SELECTION METHOD: ")
    idx = scores.index(min(scores))
    f.write(methods[idx]+'\n')
    f.write("SELECTED FEATURES:\n")
    f.write(str(names))
    f.write('---------------------------------------\n')

def write_model_results(f, name, models, scores):
    f.write(name+'\n')
    for i in range(len(scores)):
        f.write(models[i]+'\t\t'+str(scores[i])+'\n')
    f.write("BEST SELECTED MODEL: ")
    idx = scores.index(min(scores))
    f.write(models[idx]+'\n')
    f.write('---------------------------------------\n')