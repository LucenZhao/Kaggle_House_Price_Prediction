from pandas import read_csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def visualize(data, label):

    plt.subplot(221)
    plt.scatter(label, data['1stFlrSF'])
    plt.ylabel('1stFlrSF')
    plt.subplot(222)
    plt.scatter(label, data['YearBuilt'])
    plt.ylabel('YearBuilt')
    plt.subplot(223)
    plt.scatter(label, data['TotalBsmtSF'])
    plt.ylabel('TotalBsmtSF')
    plt.subplot(224)
    plt.scatter(label, data['GrLivArea'])
    plt.ylabel('GrLivArea')
    plt.show()

def load_data(train_path, test_path):
    
    # read data
    df = read_csv(train_path)
    df_test = read_csv(test_path)

    # get index, target and predictors
    idx = df['Id']  # index
    test_idx = df_test['Id']
    targets = df['SalePrice']  # target
    targets = np.log(targets)
    x_cols = list(df.loc[:,'MSSubClass':'SaleCondition'])  # predictors
    x = df[x_cols]
    test_x = df_test[x_cols]

    # eliminate or replace nan values
    x = x.dropna(axis=1, how='any', thresh=1000)
    names = list(x)
    objs = (x.dtypes == object)
    obj_idx = list(x.loc[:, x.dtypes == object])
    x[obj_idx] = x[obj_idx].fillna('others')
    num_idx = list(np.where(x.isna().any())[0])
    for col in num_idx:
        idx = list(x)[col]
        mean = np.mean(x[idx])
        x[idx] = x[idx].fillna(mean)

    # same operation on testing dataset
    test_x = test_x[names]
    test_x[obj_idx] = test_x[obj_idx].fillna('others')
    num_idx = list(np.where(test_x.isna().any())[0])
    for col in num_idx:
        idx = list(test_x)[col]
        mean = np.mean(test_x[idx])
        test_x[idx] = test_x[idx].fillna(mean)

    # category encoding and normalization
    le = LabelEncoder()
    for i, col in x.iteritems():
        if objs[i] == True:
            le.fit(col)
            x[i] = le.transform(x[i])
            test_x[i] = le.transform(test_x[i])
        else:
            mean = np.mean(col)
            x[i] = x[i] / mean
            test_x[i] = test_x[i] / mean

    return x, targets, idx, df, test_x, test_idx

if __name__ == '__main__':
	import config
	x, label, _, data, _, _ = load_data(config.data_path)
	visualize(data, label)
