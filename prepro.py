from pandas import read_csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def analysis(data, label):
    sex = {'maleY': 0, 'maleN': 0, 'femaleY': 0, 'femaleN': 0}
    embarked = {'Y1': 0, 'N1': 0, 'Y2': 0, 'N2': 0, 'Y3': 0, 'N3': 0}
    pclass = {'Y1': 0, 'N1': 0, 'Y2': 0, 'N2': 0, 'Y3': 0, 'N3': 0}
    print(data)
    for v in data.iterrows():
        print(v)
        v = v[1]
        if v['Survived'] == 0:
            if v['Sex'] == 1:
                sex['maleN'] += 1
            else:
                sex['femaleN'] += 1

            if v['Pclass'] == 1:
                pclass['N1'] += 1
            elif v['Pclass'] == 2:
                pclass['N2'] += 1
            else:
                pclass['N3'] += 1

            if v['Embarked'] == 1:
                embarked['N1'] += 1
            elif v['Embarked'] == 2:
                embarked['N2'] += 1
            else:
                embarked['N3'] += 1

        else:
            if v['Sex'] == 1:
                sex['maleY'] += 1
            else:
                sex['femaleY'] += 1

            if v['Pclass'] == 1:
                pclass['Y1'] += 1
            elif v['Pclass'] == 2:
                pclass['Y2'] += 1
            else:
                pclass['Y3'] += 1

            if v['Embarked'] == 1:
                embarked['Y1'] += 1
            elif v['Embarked'] == 2:
                embarked['Y2'] += 1
            else:
                embarked['Y3'] += 1

    print(sex)
    print(embarked)
    print(pclass)

def visualize(data, label):
#    dataY = data[label == 1]
#    dataN = data[label == 0]
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
#    fig, ax = plt.subplots()
#    ax.boxplot([dataY['Age'], dataN['Age']])
#    plt.show()
#    fig, ax = plt.subplots()
#    ax.boxplot([dataY['Fare'], dataN['Fare']])
#    plt.show()
#    fig, ax = plt.subplots()
#    ax.boxplot([dataY['Parch'], dataN['Parch']])
#    plt.show()
#    fig, ax = plt.subplots()
#    ax.boxplot([dataY['SibSp'], dataN['SibSp']])
#    plt.show()
    plt.show()

def load_test_data(path):
    # read data
    df = read_csv(path)

    data_idx = df['Id']
#    print(df.shape)
#    print(df.dtypes)
#    print(sum(df.dtypes == object))
#    print(sum(df.dtypes == np.int64))
#    print(sum(df.dtypes == np.float64))

    # eliminate or replace nan values
    objs = (df.dtypes == object)
    obj_idx = list(df.loc[:, df.dtypes == object])
    df[obj_idx] = df[obj_idx].fillna('others')
    num_idx = list(np.where(df.isna().any())[0])
    for col in num_idx:
        idx = list(df)[col]
        mean = np.mean(df[idx])
        df[idx] = df[idx].fillna(mean)
#    df = df.dropna(axis=1, how='any')
    # category encoding and normalization
    print(df.shape)
    le = LabelEncoder()
    for i, col in df.iteritems():
        if objs[i] == True:
            df[i] = le.fit_transform(col)
        else:
            mean = np.mean(col)
            df[i] = col / mean

#    for i, v in enumerate(data['Age']):
#        if math.isnan(v):
#            data['Age'][i] = mean_age

#    for i, v in enumerate(data['Fare']):
#        if math.isnan(v):
#            data['Fare'][i] = mean_fare

#    for i, v in enumerate(data['Embarked']):
#        if math.isnan(v):
#            data['Embarked'][i] = 1 
#    data['Age'] = data['Age'].replace('NaN', mean_age)

    # get targets and predictors
#    targets = df['SalePrice']
#    targets = np.log(targets)
    
    x_cols = list(df.loc[:,'MSSubClass':'SaleCondition'])
    x = df[x_cols]

    return x, data_idx, df
def load_data(path):
    
    # read data
    df = read_csv(path)
    idx = df['Id']
    targets = df['SalePrice']
    targets = np.log(targets)
#    print(df.shape)
#    print(df.dtypes)
#    print(sum(df.dtypes == object))
#    print(sum(df.dtypes == np.int64))
#    print(sum(df.dtypes == np.float64))

    # eliminate or replace nan values
    df = df.dropna(axis=1, how='any', thresh=1000)
    print(list(df))
    objs = (df.dtypes == object)
    obj_idx = list(df.loc[:, df.dtypes == object])
    df[obj_idx] = df[obj_idx].fillna('others')
    num_idx = list(np.where(df.isna().any())[0])
    for col in num_idx:
        idx = list(df)[col]
        mean = np.mean(df[idx])
        df[idx] = df[idx].fillna(mean)
    df = df.dropna(axis=1, how='any')
    # category encoding and normalization
    print(df.shape)
    le = LabelEncoder()
    for i, col in df.iteritems():
        if objs[i] == True:
            df[i] = le.fit_transform(col)
        else:
            mean = np.mean(col)
            df[i] = col / mean

#    for i, v in enumerate(data['Age']):
#        if math.isnan(v):
#            data['Age'][i] = mean_age

#    for i, v in enumerate(data['Fare']):
#        if math.isnan(v):
#            data['Fare'][i] = mean_fare

#    for i, v in enumerate(data['Embarked']):
#        if math.isnan(v):
#            data['Embarked'][i] = 1 
#    data['Age'] = data['Age'].replace('NaN', mean_age)

    # get targets and predictors

    
    x_cols = list(df.loc[:,'MSSubClass':'SaleCondition'])
    x = df[x_cols]

    return x, targets, idx, df

if __name__ == '__main__':
	import config
	x, label, _, data = load_data(config.data_path)
#	analysis(data, label)
	visualize(data, label)
