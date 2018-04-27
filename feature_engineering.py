import numpy as np
import config
from sklearn.feature_selection import SelectPercentile, f_regression, 
                                      mutual_info_regression, RFECV


def construct_features(train_X, train_Y, test_X, have_poly=True):
    
    # find the most important features
    sel = SelectPercentile(f_regression, percentile=20)
    sel.fit(train_X, train_Y)
    sup = sel.get_support()

    sel_idx = np.where(sup == True)[0]
    sel_names = [names[i] for i in sel_idx]
    sel_names = [n for i,n in enumerate(sel_names) if floats[sel_idx[i]] == True]
    
    # feature construction
    d = {}  # training
    d_t = {}  # testing

    # construct features by combining 2 different features
    for i, n1 in enumerate(sel_names):
        for j, n2 in enumerate(sel_names):
            if i != j:
                new_col = train_X[n1] * train_X[n2]
                new_col_t = test_X[n1] * test_X[n2]
                new_name = n1 + '*' + n2
                d[new_name] = new_col
                d_t[new_name] = new_col_t
    comb_X = pandas.DataFrame(data=d)
    comb_X_t = pandas.DataFrame(data=d_t)

    if have_poly is False:
        new_X = train_X.join(comb_X)
        new_test = test_X.join(comb_X_t)
        return new_X, new_test

    # construct features by making polynimial terms
    float_names = [n for i,n in enumerate(names) if floats[i] == True]
    quad_X = train_X[float_names] ** 2
    quad_X_t = test_X[float_names] ** 2
    quad_X.columns = [n + '^2' for n in float_names]
    quad_X_t.columns = [n + '^2' for n in float_names]
    tri_X = train_X[float_names] ** 3
    tri_X_t = test_X[float_names] ** 3
    tri_X.columns = [n + '^3' for n in float_names]
    tri_X_t.columns = [n + '^3' for n in float_names]
    poly_X = quad_X.join(tri_X)
    poly_X_t = quad_X_t.join(tri_X_t)
    comb_X = comb_X.join(poly_X)
    comb_X_t = comb_X_t.join(poly_X_t)
    new_X = train_X.join(comb_X)
    new_test = test_X.join(comb_X_t)
    
    return new_X, new_test

def select_features(train_X, train_Y, test_X, models, features):
    
    # feature construction
    new_X, new_test = construct_features(train_X, train_Y, test_X)
    new_names = list(new_X)
    # selection models
    sel1 = SelectPercentile(f_regression, percentile=80)
    sel2 = SelectPercentile(mutual_info_regression, percentile=80)
    pca = PCA(n_components=80)

    scores = []
    best_feats = []
    best_test = []
    best_score = 1
    best_names = []
    for i, m in enumerate(features):
        if m == 'base':
            feats = train_X
            test = test_X
            names = list(feats)
        if m == 'construct':
            feats = new_X
            test = new_test
            names = list(feats)
        if m == 'freg':
            feats = sel1.fit_transform(new_X)
            test = sel1.transform(new_test)
            sup = sel1.get_support()
            sel_idx = np.where(sup == True)[0]
            names = [new_names[i] for i in sel_idx]
        if m == 'mutinfo':
            feats = sel2.fit_transform(new_X)
            test = sel2.transform(new_test)
            sup = sel2.get_support()
            sel_idx = np.where(sup == True)[0]
            names = [new_names[i] for i in sel_idx]
        if m == 'pca':
            feats = pca.fit_transform(new_X)
            test = pca.transform(new_test)
            sup = pca.get_support()
            sel_idx = np.where(sup == True)[0]
            names = [new_names[i] for i in sel_idx]
        if m == 'rfe':
            rfe = RFECV(models[i])
            feats = rfe.fit_transform(new_X)
            test = rfe.transform(new_test)
            sup = rfe.get_support()
            sel_idx = np.where(sup == True)[0]
            names = [new_names[i] for i in sel_idx]

        k_fold = KFold(n_splits=config.cv_folds)
        score = sum(cross_val_score(models[i], feats, train_Y, cv=k_fold,
            scoring='neg_mean_squared_error')) / config.cv_folds
        scores.append(-1 * score)

        if score < best_score:
            best_score = score
            best_feats = feats
            best_names = names
            best_test = test
    
    scores = np.sqrt(scores)
    return scores, best_feats, best_test, best_names
