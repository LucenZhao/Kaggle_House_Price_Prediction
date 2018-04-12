import config
from prepro import load_data, load_test_data
from utils import cross_valid, write_test_file
import numpy as np
import pandas
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
                                          QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, explained_variance_score
from pygam import LinearGAM
from pygam.utils import generate_X_grid

def main():
    
    # pre-process data
    train_X, train_Y, train_idx, _ = load_data(config.data_path)
    test_X, test_idx, _ = load_test_data(config.test_path)
    print(test_idx)
    names = list(train_X)
    types = train_X.dtypes
    floats = (types == np.float64)

#    lasso1 = Lasso(alpha=0.0001)
    lasso = Lasso(alpha=0.001)

#    lasso3 = Lasso(alpha=0.01)
#    lasso4 = Lasso(alpha=0.05)
#    lasso5 = Lasso(alpha=0.1)

#    ridge1 = Ridge(alpha=0.01)
#    ridge2 = Ridge(alpha=0.1)
#    ridge3 = Ridge(alpha=1)
#    ridge4 = Ridge(alpha=5)
#    ridge5 = Ridge(alpha=10)
#    kn3 = KNeighborsRegressor(n_neighbors=3, weights='distance')
#    kn5 = KNeighborsRegressor(n_neighbors=5, weights='distance')
#    kn10 = KNeighborsRegressor(n_neighbors=10, weights='distance')
#    
#    rf1 = RandomForestRegressor(n_estimators=500, max_depth=4, max_features='auto')
#    rf2 = RandomForestRegressor(n_estimators=500, max_depth=4, max_features='sqrt')
#    rf3 = RandomForestRegressor(n_estimators=500, max_depth=4, max_features='log2')
#    rf4 = RandomForestRegressor(n_estimators=1000, max_depth=4, max_features='auto')
#    rf5 = RandomForestRegressor(n_estimators=1000, max_depth=4, max_features='sqrt')
#    rf6 = RandomForestRegressor(n_estimators=1000, max_depth=4, max_features='log2')
    est = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.01,
          max_depth=1, random_state=0, loss='ls') 
#    est2 = GradientBoostingRegressor(n_estimators=500, learning_rate=0.001,
#          max_depth=1, random_state=0, loss='ls') 
#    est3 = GradientBoostingRegressor(n_estimators=700, learning_rate=0.01,
#          max_depth=1, random_state=0, loss='ls') 
#    est4 = GradientBoostingRegressor(n_estimators=700, learning_rate=0.001,
#          max_depth=1, random_state=0, loss='ls') 
#    est5 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.01,
#          max_depth=1, random_state=0, loss='ls') 
#    est6 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.001,
#          max_depth=1, random_state=0, loss='ls') 
    '''
    rf = rf.fit(train_X, train_Y)
    selected = np.where(rf.feature_importances_ > 1e-2)
    selected = list(selected[0])
    for s in selected:
        print(names[s])
    sfm1 = SelectFromModel(rf, prefit=True, threshold=1e-3)
    X_new1 = sfm1.transform(train_X)
    
    sfm2 = SelectFromModel(ridge3, threshold=5e-2)
    sfm2.fit(train_X, train_Y)
    X_new2 = sfm2.transform(train_X)
    sup1 = sfm2.get_support()
    print(train_X.shape)
    print(X_new2)
    print(sup1)
    '''
    sel = SelectPercentile(f_regression, percentile=20)
    sel.fit(train_X, train_Y)
    sup = sel.get_support()

    sel_idx = np.where(sup == True)[0]
    sel_names = [names[i] for i in sel_idx]
    sel_names = [n for i,n in enumerate(sel_names) if floats[sel_idx[i]] == True]
    
    d = {}
    d_t = {}
    for i,n1 in enumerate(sel_names):
        for j,n2 in enumerate(sel_names):
            if i != j:
                new_col = train_X[n1] * train_X[n2]
                new_col_t = test_X[n1] * test_X[n2]
                new_name = n1 + '*' + n2
                d[new_name] = new_col
                d_t[new_name] = new_col_t
    comb_X = pandas.DataFrame(data=d)
    comb_X_t = pandas.DataFrame(data=d_t)
#    print(comb_X.shape)
    float_names = [n for i,n in enumerate(names) if floats[i] == True]
    quad_X = train_X[float_names] ** 2
    quad_X_t = test_X[float_names] ** 2
    quad_X.columns = [n + '^2' for n in float_names]
    quad_X_t.columns = [n + '^2' for n in float_names]
    tri_X = train_X[float_names] ** 3
    tri_X_t = test_X[float_names] ** 3
    tri_X.columns = [n + '^3' for n in float_names]
    tri_X_t.columns = [n + '^3' for n in float_names]
#    quat_X = train_X[sel_names] ** 4
#    quat_X.columns = [n + '^4' for n in sel_names]
    poly_X = quad_X.join(tri_X)
    poly_X_t = quad_X_t.join(tri_X_t)
    comb_X = comb_X.join(poly_X)
    comb_X_t = comb_X_t.join(poly_X_t)
#    poly_X = poly_X.join(quat_X)
#    print(poly_X)
    new_X = train_X.join(comb_X)
    new_test = test_X.join(comb_X_t)
    print(new_X.shape)
    print(new_test.shape)
    new_names = list(new_X)
#    poly = PolynomialFeatures(2)
#    new_X = poly.fit_transform(train_X)
#    print(new_X.shape)
#    models = [lasso1, lasso2, lasso3, lasso4, lasso5, ridge1, ridge2, ridge3, ridge4, ridge5, kn5, rf, est]
#    models = [lasso1, lasso3, ridge1, ridge2, kn5, rf, est]
#    models = [lasso1, lasso2, lasso3, lasso4, lasso5]
#    models = [est1]
#    scores1 = cross_valid(models, new_X, train_Y)
#    scores2 = cross_valid(models, train_X, train_Y)

    rfe = RFECV(lasso, cv=3)
#    rfe2 = RFECV(lasso2, cv=3)
#    rfe3 = RFECV(lasso3, cv=3)
#    rfe4 = RFECV(lasso4, cv=3)
#    rfe5 = RFECV(lasso5, cv=3)
#    rfe3 = RFE(rf, 150)
#    rfe4 = RFE(est, 150)
    sel = SelectPercentile(mutual_info_regression, percentile=80)
#    sel2 = SelectPercentile(f_regression, percentile=80)
#    pca1 = PCA(n_components=50)
    pca = PCA(n_components=90)
    new_X = sel.fit_transform(new_X, train_Y)
    new_test = sel.transform(new_test)
    sup = sel.get_support()
    sel_idx = np.where(sup == True)[0]
    sel_names = [new_names[i] for i in sel_idx]
    print(sel_names)
#    new1_X = rfe1.fit_transform(new_X, train_Y)
    print(new_X.shape)

#    valid_X = new_X[:200]
    valid_Y = train_Y[:200] 
#    new_X = new_X[200:]
    train1_Y = train_Y[200:]

#    est.fit(new_X, train_Y)
#    preds = est.predict(new_test)
    err = []
    for i in range(90, 100, 10):
        sel = SelectPercentile(mutual_info_regression, percentile=i)
        new1_X = sel.fit_transform(new_X, train_Y)
        valid_X = new1_X[:200]
 
        train_X = new1_X[200:]

        est.fit(train_X, train1_Y)
        predictions = est.predict(valid_X)
#    preds = np.exp(predictions)
#    print(predictions)
#    print(preds)
#    write_test_file(preds, test_idx)
        err.append(np.sqrt(mean_squared_error(valid_Y, predictions)))
        print(explained_variance_score(valid_Y, predictions))
        print(r2_score(valid_Y, predictions))
        plt.scatter(valid_Y, predictions)
        x = [10.5, 11, 11.5, 12, 12.5, 13, 13.5]
        y = [10.5, 11, 11.5, 12, 12.5, 13, 13.5]
        plt.plot(x,y,'--')
        plt.ylabel("Predictions")
        plt.xlabel("Actual Y-values")
        plt.show()
#    plt.plot([10,20,30,40,50,60,70,80,90],err)
#    plt.xlabel("Percentage of Feature")
#    plt.ylabel("Validation MSE")
#    plt.show()
#    preds = np.exp(preds)
#    write_test_file(preds, test_idx)
#    new2_X = rfe2.fit_transform(new_X, train_Y)
#    print(new2_X.shape)
#    new3_X = rfe3.fit_transform(new_X, train_Y)
#    print(new3_X.shape)
#    new4_X = rfe4.fit_transform(new_X, train_Y)
#    print(new4_X.shape)
#    new5_X = rfe5.fit_transform(new_X, train_Y)
#    print(new5_X.shape)
#    new2_X = rfe2.fit_transform(new_X, train_Y)
#    new1_X = rfe1.fit_transform(new_X, train_Y)
#    new2_X = rfe2.fit_transform(new_X, train_Y)
#    new3_X = rfe3.fit_transform(new_X, train_Y)
#    new4_X = rfe4.fit_transform(new_X, train_Y)
#    pca1.fit(train_X, train_Y)
#    sel3.fit(train_X, train_Y)
#    new4_X = pca2.fit_transform(new_X, train_Y)

#    names1 = [new_names[i] for i in np.where(rfe1.support_ == True)[0]]
#    names2 = [new_names[i] for i in np.where(rfe2.support_ == True)[0]]

#    scores1 = cross_valid(models, new_X, train_Y)
#    scores2 = cross_valid([lasso2], new2_X, train_Y)
#    scores3 = cross_valid([lasso3], new3_X, train_Y)
#    scores4 = cross_valid([lasso4], new4_X, train_Y)
#    scores5 = cross_valid([lasso5], new5_X, train_Y)
#    scores5 = cross_valid(models, new3_X, train_Y)
#    scores5 = cross_valid(models, new3_X, train_Y)
#    scores6 = cross_valid(models, new4_X, train_Y)
#    print(sel_names)
#    print(new1_X.shape)
#    print(new2_X.shape)
#    print(new_X.shape)
#    print(scores1)
#    print(scores2)
#    print(scores3)
#    print(scores4)
#    print(scores5)
#    valid_X = new_X[:200]
    valid_Y = train_Y[:200] 
#    train_X = new_X[200:]
    train_Y = train_Y[200:]
#    new_train = sel3.transform(train_X)
#    new_valid = sel3.transform(valid_X)
#    print(new_valid.shape)
    err = []
    for i in range(80, 90, 10):
        pca = PCA(n_components=i)
        new1_X = pca.fit_transform(new_X, train_Y)
        valid_X = new1_X[:200]
 
        train_X = new1_X[200:]

        gam = LinearGAM(n_splines=8).gridsearch(train_X, train_Y)
        predictions = gam.predict(valid_X)
#    preds = np.exp(predictions)
#    print(predictions)
#    print(preds)
#    write_test_file(preds, test_idx)
        err.append(np.sqrt(mean_squared_error(valid_Y, predictions)))
        print(explained_variance_score(valid_Y, predictions))
        print(r2_score(valid_Y, predictions))
        plt.scatter(valid_Y, predictions)
        x = [10.5, 11, 11.5, 12, 12.5, 13, 13.5]
        y = [10.5, 11, 11.5, 12, 12.5, 13, 13.5]
        plt.plot(x,y,'--')
        plt.ylabel("Predictions")
        plt.xlabel("Actual Y-values")
        plt.show()
#    print(gam.summary())
#    print(scores5)
#    print(scores6)
if __name__ == '__main__':
	main()