import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from time import time
# import random
# from backtest import backtest, backtestData
# from backtest1 import backtest1, backtestData1

# input the raw data
path = os.getcwd() + '\\MR raw data\\'
df_all = pd.read_csv(path + 'df_all_close_price.csv', converters={'stcode': str})
# timelist_m = pd.read_excel(path + 'timelist_m.xlsx', header=None, squeeze=True)

# print(df_all.head())
# print(timelist_m.head())

# set up the training group and the testing group according to year
df_train_na0 = df_all[(df_all['trade_dt'] <= '2017-12-31') | (df_all['trade_dt'] >= '2020-01-01')].reset_index(drop=True)
df_test_na0 = df_all[(df_all['trade_dt'] > '2017-12-31') & (df_all['trade_dt'] < '2020-01-01')].reset_index(drop=True)

df_train_0 = df_train_na0.dropna(axis=0, how='any')
df_test_0 = df_test_na0.dropna(axis=0, how='any')

df_train = df_train_0[df_train_0['flag'] != 0]
df_test = df_test_0.copy()
# df_test = df_test_0[df_test_0['flag'] != 0]

# print(df_train.head())
# print(df_test.head())

# set up the datasets
factor_list = ['规模', '估值', '分红', '盈利', '财务质量', '成长', '反转', '波动率', '流动性', '分析师预期变化']

y_train = df_train['flag']
X_train = df_train[factor_list]
X_train = (X_train - X_train.mean()) / X_train.std()

y_test = df_test['flag']
X_test = df_test[factor_list]
X_test = (X_test - X_test.mean()) / X_test.std()

# output price table
stcode = df_test['stcode'].drop_duplicates()
price = pd.pivot_table(df_test,index=['trade_dt'],columns=['stcode'],values=['post_adjust_close'])
# print(price.columns)
stcode_sort = list(stcode)
stcode_sort.sort()
# print(stcode_sort)
price.columns = stcode_sort
price.to_csv(os.getcwd() + '\\MR result data\\' + 'price.csv')
# print(price.head())

# train the logistic regression model
model_LR = LogisticRegression(random_state=0, solver='saga', C=1, penalty='l1')
model_LR.fit(X_train, y_train)

coef_LR = pd.DataFrame(model_LR.coef_[0].tolist(), columns=['coefficient'])
coef_LR.index = factor_list
# print(coef_LR)

# print("out-of-sample accuracy: ", model_LR.score(X_test, y_test))

prob = model_LR.predict_proba(X_test)
prob_df = pd.DataFrame(prob)
if model_LR.classes_[0] == 1:
    prob_df.columns = ['1','-1']
else:
    prob_df.columns = ['-1','1']
# print(prob_df)

# present the result of logistic regression
result_LR = df_test[['trade_dt','stcode','flag']].reset_index(drop=True)
result_LR['prob'] = prob_df['1']
result_LR['pred_flag'] = result_LR.apply(lambda row: 1 if row[3] >= 0.5 else -1, axis=1)
result_LR = result_LR[['trade_dt','stcode','flag','pred_flag','prob']]
result_LR['correct'] = result_LR.apply(lambda row: 1 if row[2] == row[3] else 0, axis=1)
result_LR = result_LR.sort_values(by=['trade_dt','prob']).reset_index(drop=True)
# print(result_LR)
result_LR.to_csv(os.getcwd() + '\\MR result data\\' + 'result_LR.csv')

# calculate the accuracy
acc_LR = pd.pivot_table(result_LR,index=['trade_dt'],columns=['stcode'],values=['correct'])
stcode = result_LR['stcode'].drop_duplicates().to_list()
stcode.sort()
acc_LR.columns = stcode
# print(acc_LR)

accuracy_LR = acc_LR.sum(axis=1)/acc_LR.count(axis=1)
# print(accuracy_LR)

# draw the figure
fig, ax1 = plt.subplots(figsize=(18, 8))
plt.plot(accuracy_LR.index.to_list(),accuracy_LR.tolist())
plt.xticks(rotation=45)
plt.title('The Accuracy of the Logistic Regression Model')
plt.savefig(os.getcwd() + '\\MR result fig\\' + 'The Accuracy of the Logistic Regression Model.png')
plt.show()

# train the Naive Bayes Bernoulli
model_NB =  BernoulliNB()
model_NB.fit(X_train,y_train)

# print("out-of-sample accuracy: ", model_NB.score(X_test,y_test))

prob = model_NB.predict_proba(X_test)
prob_df = pd.DataFrame(prob)
if model_NB.classes_[0] == 1:
    prob_df.columns = ['1','-1']
else:
    prob_df.columns = ['-1','1']
# print(prob_df)

# present the result of Naive Bayes Bernoulli
result_NB = df_test[['trade_dt','stcode','flag']].reset_index(drop=True)
result_NB['prob'] = prob_df['1']
result_NB['pred_flag'] = result_NB.apply(lambda row: 1 if row[3] >= 0.5 else -1, axis=1)
result_NB = result_NB[['trade_dt','stcode','flag','pred_flag','prob']]
result_NB['correct'] = result_NB.apply(lambda row: 1 if row[2] == row[3] else 0, axis=1)
result_NB = result_NB.sort_values(by=['trade_dt','prob']).reset_index(drop=True)
# print(result_NB)
result_NB.to_csv(os.getcwd() + '\\MR result data\\' + 'result_NB.csv')

# calculate the accuracy
acc_NB = pd.pivot_table(result_NB,index=['trade_dt'],columns=['stcode'],values=['correct'])
stcode = result_NB['stcode'].drop_duplicates().to_list()
stcode.sort()
acc_NB.columns = stcode
# print(acc_NB)

accuracy_NB = acc_NB.sum(axis=1)/acc_NB.count(axis=1)
# print(accuracy_NB)

# draw the figure
fig, ax1 = plt.subplots(figsize=(18, 8))
plt.plot(accuracy_NB.index.to_list(),accuracy_NB.tolist())
plt.xticks(rotation=45)
plt.title('The Accuracy of the Naive Bayes Bernoulli Model')
plt.savefig(os.getcwd() + '\\MR result fig\\' + 'The Accuracy of the Naive Bayes Bernoulli Model.png')
plt.show()

# train the Gradient Boosting Classifier
model_GB =  GradientBoostingClassifier(random_state=0, n_estimators=300, max_depth=4, min_samples_split=120, min_samples_leaf=30)
model_GB.fit(X_train,y_train)

ipt_GB = pd.DataFrame(model_GB.feature_importances_.tolist(), columns=['feature_importance'])
ipt_GB.index = factor_list
# print(ipt_GB)

# print("out-of-sample accuracy: ", model_GB.score(X_test,y_test))

prob = model_GB.predict_proba(X_test)
prob_df = pd.DataFrame(prob)
if model_GB.classes_[0] == 1:
    prob_df.columns = ['1','-1']
else:
    prob_df.columns = ['-1','1']
# print(prob_df)

# present the result of Gradient Boosting Classifier
result_GB = df_test[['trade_dt','stcode','flag']].reset_index(drop=True)
result_GB['prob'] = prob_df['1']
result_GB['pred_flag'] = result_GB.apply(lambda row: 1 if row[3] >= 0.5 else -1, axis=1)
result_GB = result_GB[['trade_dt','stcode','flag','pred_flag','prob']]
result_GB['correct'] = result_NB.apply(lambda row: 1 if row[2] == row[3] else 0, axis=1)
result_GB = result_GB.sort_values(by=['trade_dt','prob']).reset_index(drop=True)
# print(result_GB)
result_GB.to_csv(os.getcwd() + '\\MR result data\\' + 'result_GB.csv')

# calculate the accuracy
acc_GB = pd.pivot_table(result_GB,index=['trade_dt'],columns=['stcode'],values=['correct'])
stcode = result_GB['stcode'].drop_duplicates().to_list()
stcode.sort()
acc_GB.columns = stcode
# print(acc_GB)

accuracy_GB = acc_GB.sum(axis=1)/acc_GB.count(axis=1)
# print(accuracy_GB)

# draw the figure
fig, ax1 = plt.subplots(figsize=(18, 8))
plt.plot(accuracy_GB.index.to_list(),accuracy_GB.tolist())
plt.xticks(rotation=45)
plt.title('The Accuracy of the Gradient Boosting Classifier Model')
plt.savefig(os.getcwd() + '\\MR result fig\\' + 'The Accuracy of the Gradient Boosting Classifier Model.png')
plt.show()