import config
from prepro import load_data, load_test_data
from feature_engineering import select_features, construct_features 
from utils import cross_valid, write_test_file, write_selection_results, write_model_results
import numpy as np
import pandas
import matplotlib.pyplot as plt


from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score, explained_variance_score
from pygam import LinearGAM
from pygam.utils import generate_X_grid

def main():
    
    f = open('results.txt', 'w')

    f.write("Preprocessing data...\n\n")
    # pre-process data
    train_X, train_Y, train_idx, _, test_X, test_idx = load_data(config.data_path, config.test_path)
    names = list(train_X)
    types = train_X.dtypes
    floats = (types == np.float64)

    new_X_GAM, new_test_GAM = construct_features(train_X, train_Y, test_X, have_poly=False)
    
    # feature selection
    f.write("Feature Selection\n")
    ridge_scores, ridge_X, ridge_test, ridge_names = select_features(train_X, train_Y, test_X, config.ridge_select, config.ridge_feats)
    lasso_scores, lasso_X, lasso_test, lasso_names = select_features(train_X, train_Y, test_X, config.lasso_select, config.lasso_feats)
    knn_scores, knn_X, knn_test, knn_names = select_features(train_X, train_Y, test_X, config.knn_select, config.knn_feats)
    rf_scores, rf_X, rf_test, rf_names = select_features(train_X, train_Y, test_X, config.rf_select, config.rf_feats)
    est_scores, est_X, est_test, est_names = select_features(train_X, train_Y, test_X, config.est_select, config.est_feats)
    write_selection_results(f, 'Ridge Regression', config.ridge_feats, ridge_scores, ridge_names)
    write_selection_results(f, 'LASSO Regression', config.lasso_feats, lasso_scores, lasso_names)
    write_selection_results(f, 'K-Nearest Neighbours', config.knn_feats, knn_scores, knn_names)
    write_selection_results(f, 'Random Forest', config.rf_feats, rf_scores, rf_names)
    write_selection_results(f, 'Gradient Boosting', config.est_feats, est_scores, est_names)
    f.write('\n#######################################\n\n')

    # model selection
    f.write("Model Selection\n")
    ridge_scores = cross_valid(config.ridge_models, ridge_X, train_Y)
    lasso_scores = cross_valid(config.lasso_models, lasso_X, train_Y)
    knn_scores = cross_valid(config.knn_models, knn_X, train_Y)
    rf_scores = cross_valid(config.rf_models, rf_X, train_Y)
    est_scores = cross_valid(config.est_models, est_X, train_Y)
    write_model_results(f, 'Ridge Regression', config.ridge_models, ridge_scores)
    write_model_results(f, 'LASSO Regression', config.lasso_models, lasso_scores)
    write_model_results(f, 'K-Nearest Neighbours', config.knn_models, knn_scores)
    write_model_results(f, 'Random Forest', config.rf_models, rf_scores)
    write_model_results(f, 'Gradient Boosting', config.est_models, est_scores)
    f.write('\n#######################################\n\n')

    best_reg = config.lasso3
    best_tree = config.est3
    best_reg.fit(lasso_X, train_Y)
    predictions_reg = best_reg.predict(lasso_test)
    best_tree.fit(est_X, train_Y)
    predictions_tree = best_tree.predict(est_test)
    write_test_file(predictions_reg, test_idx, 'results_reg.csv')
    write_test_file(predsictions_tree, test_idx, 'results_tree.csv')

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