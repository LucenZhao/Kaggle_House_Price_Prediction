from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA

data_path = "data/train.csv"
test_path = 'data/test.csv'
freg_perc = 80
mutinfo_perc = 80
pca_perc = 80
cv_folds = 3

ridge1 = Ridge(alpha=0.0001)
ridge2 = Ridge(alpha=0.001)
ridge3 = Ridge(alpha=0.01)
ridge4 = Ridge(alpha=0.1)
ridge5 = Ridge(alpha=1)
ridge6 = Ridge(alpha=3)
ridge_select = [ridge1, ridge1, ridge1, ridge1, ridge5, ridge2]
ridge_feats = ['base', 'construct', 'freg', 'mutinfo', 'pca', 'rfe']
ridge_models = [ridge2, ridge3, ridge4, ridge5, ridge6]

lasso1 = Lasso(alpha=0.00001)
lasso2 = Lasso(alpha=0.0001)
lasso3 = Lasso(alpha=0.001)
lasso4 = Lasso(alpha=0.01)
lasso5 = Lasso(alpha=0.05)
lasso6 = Lasso(alpha=0.1)
lasso_select = [lasso1, lasso1, lasso1, lasso1, lasso1, lasso1]
lasso_feats = ['base', 'construct', 'freg', 'mutinfo', 'pca', 'rfe']
lasso_models = [lasso2, lasso3, lasso4, lasso5, lasso6]

kn3 = KNeighborsRegressor(n_neighbors=3, weights='distance')
kn5 = KNeighborsRegressor(n_neighbors=5, weights='distance')
kn10 = KNeighborsRegressor(n_neighbors=10, weights='distance')
knn_select = [kn5, kn5, kn5, kn5, kn5]
knn_feats = ['base', 'construct', 'freg', 'mutinfo', 'pca']
knn_models = [kn3, kn5, kn10]

rf1 = RandomForestRegressor(n_estimators=500, max_depth=4, max_features='auto')
rf2 = RandomForestRegressor(n_estimators=500, max_depth=4, max_features='sqrt')
rf3 = RandomForestRegressor(n_estimators=500, max_depth=4, max_features='log2')
rf4 = RandomForestRegressor(n_estimators=1000, max_depth=4, max_features='auto')
rf5 = RandomForestRegressor(n_estimators=1000, max_depth=4, max_features='sqrt')
rf6 = RandomForestRegressor(n_estimators=1000, max_depth=4, max_features='log2')
rf_select = [rf1, rf1, rf1, rf1, rf1]
rf_feats = ['base', 'construct', 'freg', 'mutinfo', 'pca']
rf_models = [rf1, rf2, rf3, rf4, rf5, rf6]

est1 = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,
      max_depth=1, random_state=0, loss='ls') 
est2 = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01,
      max_depth=1, random_state=0, loss='ls') 
est3 = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.01,
      max_depth=1, random_state=0, loss='ls') 
est4 = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.001,
      max_depth=1, random_state=0, loss='ls') 
est5 = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.001,
      max_depth=1, random_state=0, loss='ls') 
est_select = [est1, est1, est1, est1, est1]
est_feats = ['base', 'construct', 'freg', 'mutinfo', 'pca']
est_models = [est1, est2, est3, est4, est5]

gam_feats = ['base', 'construct', 'freg', 'mutinfo', 'pca']


