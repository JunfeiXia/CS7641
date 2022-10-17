import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree,preprocessing
from sklearn import svm,neighbors
from sklearn.model_selection import validation_curve,train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# dataset 1
# filename = 'Algerian_forest_fires_dataset_clean.csv' # dataset 1
# Attributes = ["Temperature","RH","Ws","Rain"]
# Target_names = ["not fire","fire"]
# Label_feature_name=["class_label_index"]

# dataset 2
filename = 'Occupancy_Estimation_clean.csv' # dataset 2
Attributes = ["Average Temp","Average Light","Average Sound","CO2_Slope"]
Target_names = ['0','1','2','3']
Target_names1 = ['Occupied','Not Occupied']
Label_feature_name=['Room_Occupancy_Count']


def getdataX(data,Attributes:list):
    X = data.loc[:,Attributes].to_numpy()
    return X
def getdataY(data,Label_feature_name):
    Y = data.loc[:,Label_feature_name].values.tolist()
    Y = np.reshape(Y, (len(Y),))
    return Y

def Validaiton(X,y,model,param_name,param_range,xlabel_name,title):
    subset_mask = np.isin(y, [0, 1])  # binary classification: 1 vs 2
    X, y = X[subset_mask], y[subset_mask]

    train_scores, test_scores = validation_curve(
        model,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        scoring="accuracy",
        n_jobs=2,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    print(train_scores_mean)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(xlabel_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(
        param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.show()
def Validaiton1(X,y,model,param_name,param_range,xlabel_name,title):
    subset_mask = np.isin(y, [0, 1])  # binary classification: 1 vs 2
    X, y = X[subset_mask], y[subset_mask]

    train_scores, test_scores = validation_curve(
        model,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        scoring="accuracy",
        n_jobs=2,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    print(train_scores_mean)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(xlabel_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(
        param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.plot(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.show()

data=pd.read_csv(filename).convert_dtypes()
data_train, data_test = train_test_split(data , test_size=0.25)
X_train = getdataX(data_train,Attributes)
Y_train = getdataY(data_train,Label_feature_name)
X_test = getdataX(data_test,Attributes)
Y_test = getdataY(data_test,Label_feature_name)
X = getdataX(data,Attributes)
Y = getdataY(data,Label_feature_name)
Dataset = {'data':X,'target':Y,'target_names':Target_names,'feature_names':Attributes}
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)
X_partial=X[:,[0,1]]



# SVM
Validaiton(X,Y,
           model=svm.SVC(),
           param_name='gamma',
           param_range=np.logspace(-6, -1, 5),
           xlabel_name=(r"$\gamma$"),
           title="Validation Curve with SVM",
           )
Validaiton1(X,Y,
           model=svm.SVC(),
           param_name='kernel',
           param_range=['linear', 'poly', 'rbf', 'sigmoid'],
           xlabel_name="Kernel",
           title="Validation Curve with SVM",
           )

# dTree
Validaiton(X,Y,
           model=tree.DecisionTreeClassifier(),
           param_name="ccp_alpha",
           param_range=np.linspace(0, 1, num=1000),
           xlabel_name=r"$\alpha$",
           title="Validation Curve with Decision Tree",
           )
Validaiton1(X,Y,
           model=tree.DecisionTreeClassifier(),
           param_name="max_depth",
           param_range=[0,1,2,3,4,5,6,7,8],
           xlabel_name="Maximum depth",
           title="Validation Curve with Decision Tree",
           )

# KNN
Validaiton(X,Y,
           model=neighbors.KNeighborsClassifier(),
           param_name="n_neighbors",
           param_range=[3,5,10,15,20,25,30,40,50,100,150],
           xlabel_name="n of neighbors",
           title="Validation Curve with KNN",
           )
Validaiton1(X,Y,
           model=neighbors.KNeighborsClassifier(),
           param_name="weights",
           param_range=["uniform", "distance"],
           xlabel_name="weights",
           title="Validation Curve with KNN",
           )

# Ada Boost
Validaiton(X,Y,
           model=AdaBoostClassifier(),
           param_name="n_estimators",
           param_range=[5,10,20,30,50,100,500,1000],
           xlabel_name="n of estimates",
           title="Validation Curve with Ada Boost",
           )
Validaiton1(X,Y,
           model=AdaBoostClassifier(),
           param_name="learning_rate",
           param_range=np.linspace(0,4,20),
           xlabel_name="learning rate",
           title="Validation Curve with Ada Boost",
           )

# Neural Networks
Validaiton(X_std,Y,
           model=MLPClassifier(),
           param_name="hidden_layer_sizes",
           param_range=[1,3,5,10,15,20,25,30,40,50,100,500,1000],
           xlabel_name="n of hidden layers",
           title="Validation Curve with Neural Networks",
           )
Validaiton(X_std,Y,
           model=MLPClassifier(),
           param_name="learning_rate_init",
           param_range=np.logspace(-6, 1, 20),
           xlabel_name="learning rate",
           title="Validation Curve with Neural Networks",
           )