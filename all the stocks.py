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
import random
# from backtest import backtest, backtestData
from backtest2 import backtest2, backtestData2





result_LR = pd.read_csv(os.getcwd() + '\\MR result data\\' + 'result_LR.csv')
result_NB = pd.read_csv(os.getcwd() + '\\MR result data\\' + 'result_NB.csv')
result_GB = pd.read_csv(os.getcwd() + '\\MR result data\\' + 'result_GB.csv')


# pick only one stock
def strategy_LR(data: backtestData2, result=result_LR):
    result['rank_asc'] = result.groupby(['trade_dt'])['prob'].rank(method='first', ascending=True)
    result['rank_desc'] = result.groupby(['trade_dt'])['prob'].rank(method='first', ascending=False)

    rank_asc = pd.pivot_table(result, index=['trade_dt'], columns=['stcode'], values=['rank_asc'])
    rank_desc = pd.pivot_table(result, index=['trade_dt'], columns=['stcode'], values=['rank_desc'])

    stcode = result['stcode'].drop_duplicates()
    stcode_sort = list(stcode)
    stcode_sort.sort()
    rank_asc.columns = stcode_sort
    rank_desc.columns = stcode_sort

    rank_asc[rank_asc <= 100] = 0
    rank_asc[rank_asc > 100] = np.nan
    rank_desc[rank_desc <= 100] = 1
    rank_desc[rank_desc > 100] = np.nan
    # print(rank_asc)
    # print(rank_desc)

    signal = rank_asc.combine_first(rank_desc)
    signal = signal.ffill()
    signal = signal.fillna(0)

    #     print(signal)

    return (signal)


def strategy_NB(data: backtestData2, result=result_NB):
    result['rank_asc'] = result.groupby(['trade_dt'])['prob'].rank(method='first', ascending=True)
    result['rank_desc'] = result.groupby(['trade_dt'])['prob'].rank(method='first', ascending=False)

    rank_asc = pd.pivot_table(result, index=['trade_dt'], columns=['stcode'], values=['rank_asc'])
    rank_desc = pd.pivot_table(result, index=['trade_dt'], columns=['stcode'], values=['rank_desc'])

    stcode = result['stcode'].drop_duplicates()
    stcode_sort = list(stcode)
    stcode_sort.sort()
    rank_asc.columns = stcode_sort
    rank_desc.columns = stcode_sort

    rank_asc[rank_asc <= 100] = 0
    rank_asc[rank_asc > 100] = np.nan
    rank_desc[rank_desc <= 100] = 1
    rank_desc[rank_desc > 100] = np.nan
    # print(rank_asc)
    # print(rank_desc)

    signal = rank_asc.combine_first(rank_desc)
    signal = signal.ffill()
    signal = signal.fillna(0)

    #     print(signal)

    return (signal)


def strategy_GB(data: backtestData2, result=result_GB):
    result['rank_asc'] = result.groupby(['trade_dt'])['prob'].rank(method='first', ascending=True)
    result['rank_desc'] = result.groupby(['trade_dt'])['prob'].rank(method='first', ascending=False)

    rank_asc = pd.pivot_table(result, index=['trade_dt'], columns=['stcode'], values=['rank_asc'])
    rank_desc = pd.pivot_table(result, index=['trade_dt'], columns=['stcode'], values=['rank_desc'])

    stcode = result['stcode'].drop_duplicates()
    stcode_sort = list(stcode)
    stcode_sort.sort()
    rank_asc.columns = stcode_sort
    rank_desc.columns = stcode_sort

    rank_asc[rank_asc <= 100] = 0
    rank_asc[rank_asc > 100] = np.nan
    rank_desc[rank_desc <= 100] = 1
    rank_desc[rank_desc > 100] = np.nan
    # print(rank_asc)
    # print(rank_desc)

    signal = rank_asc.combine_first(rank_desc)
    signal = signal.ffill()
    signal = signal.fillna(0)

    #     print(signal)

    return (signal)


def strategy1_LR(data: backtestData2):

    df = pd.read_csv(os.getcwd() + '\\MR result data\\' + "result_LR.csv", index_col=1)
    pred = df.loc[:, ['stcode', 'prob']]
    prices = pd.read_csv(os.getcwd() + '\\MR result data\\' + "price.csv", index_col=0)
    signal = pd.read_excel(os.getcwd() + '\\BT raw data\\' + "time.xlsx", index_col=0)
    for code in prices.columns.values:
        signal[int(code)] = np.nan
    buy = 0.7
    sell = 0.5
    max = 5

    hold = []
    for i in range(signal.index.shape[0]):
        out = []
        date = prices.index[i]
        j = 0
        while j < len(hold):
            try:
                tmp = pred.loc[date]
                tmp = tmp[tmp['stcode'] == hold[j]]['prob'].values[0]
                if tmp < 1 - sell:
                    signal.loc[date][hold[j]] = 0
                    z = hold.pop(j)
                    out.append(z)
                    continue
                else:
                    j += 1
            except:
                j += 1
        tmp = max - len(hold)
        tmp_pred = pred.loc[date].tail(tmp + max)['prob'].values
        tmp_code = pred.loc[date].tail(tmp + max)['stcode'].values
        count = 0
        for j in range(len(tmp_pred)):
            if float(tmp_pred[j]) > buy and tmp_code[j] not in hold and tmp > 0 and tmp_code[j] not in out:
                hold.append(tmp_code[j])
                count += 1
                if tmp_code[j] in signal:
                    signal.loc[date][tmp_code[j]] = 1
                else:
                    signal[tmp_code[j]] = np.nan
                    signal.loc[date][tmp_code[j]] = 1
            if count == tmp:
                break
    signal = signal.ffill()
    #     print(signal.dropna(axis=1,how='all'))
    return signal


def strategy2_LR(data: backtestData2):

    df = pd.read_csv(os.getcwd() + '\\MR result data\\' + "result_LR.csv", index_col=1)
    pred = df.loc[:, ['stcode', 'prob']]
    prices = pd.read_csv(os.getcwd() + '\\MR result data\\' + "price.csv", index_col=0)
    signal = pd.read_excel(os.getcwd() + '\\BT raw data\\' + "time.xlsx", index_col=0)
    for code in prices.columns.values:
        signal[int(code)] = np.nan
    epsilon = 0.05
    buy = 0.7
    sell = 0.5
    max = 5

    hold = []
    for i in range(signal.index.shape[0]):
        out = []
        date = prices.index[i]
        j = 0
        while j < len(hold):
            ran = random.random()
            if ran < epsilon and len(hold) != 0:
                hold.pop(random.randint(0, len(hold) - 1))
            try:
                tmp = pred.loc[date]
                tmp = tmp[tmp['stcode'] == hold[j]]['prob'].values[0]
                if tmp < 1 - sell:
                    signal.loc[date][hold[j]] = 0
                    z = hold.pop(j)
                    out.append(z)
                    continue
                else:
                    j += 1
            except:
                j += 1
        tmp = max - len(hold)
        tmp_pred = pred.loc[date].tail(tmp + max)['prob'].values
        tmp_code = pred.loc[date].tail(tmp + max)['stcode'].values
        count = 0
        for j in range(len(tmp_pred)):
            if float(tmp_pred[j]) > buy and tmp_code[j] not in hold and tmp > 0 and tmp_code[j] not in out:
                hold.append(tmp_code[j])
                count += 1
                if tmp_code[j] in signal:
                    signal.loc[date][tmp_code[j]] = 1
                else:
                    signal[tmp_code[j]] = np.nan
                    signal.loc[date][tmp_code[j]] = 1
            if count == tmp:
                break
    signal = signal.ffill()
    #     print(signal.dropna(axis=1,how='all'))
    return signal


def strategy1_GB(data: backtestData2):

    df = pd.read_csv(os.getcwd() + '\\MR result data\\' + "result_GB.csv", index_col=1)
    pred = df.loc[:, ['stcode', 'prob']]
    prices = pd.read_csv(os.getcwd() + '\\MR result data\\' + "price.csv", index_col=0)
    signal = pd.read_excel(os.getcwd() + '\\BT raw data\\' + "time.xlsx", index_col=0)
    for code in prices.columns.values:
        signal[int(code)] = np.nan
    buy = 0.7
    sell = 0.5
    max = 5

    hold = []
    for i in range(signal.index.shape[0]):
        out = []
        date = prices.index[i]
        j = 0
        while j < len(hold):
            try:
                tmp = pred.loc[date]
                tmp = tmp[tmp['stcode'] == hold[j]]['prob'].values[0]
                if tmp < 1 - sell:
                    signal.loc[date][hold[j]] = 0
                    z = hold.pop(j)
                    out.append(z)
                    continue
                else:
                    j += 1
            except:
                j += 1
        tmp = max - len(hold)
        tmp_pred = pred.loc[date].tail(tmp + max)['prob'].values
        tmp_code = pred.loc[date].tail(tmp + max)['stcode'].values
        count = 0
        for j in range(len(tmp_pred)):
            if float(tmp_pred[j]) > buy and tmp_code[j] not in hold and tmp > 0 and tmp_code[j] not in out:
                hold.append(tmp_code[j])
                count += 1
                if tmp_code[j] in signal:
                    signal.loc[date][tmp_code[j]] = 1
                else:
                    signal[tmp_code[j]] = np.nan
                    signal.loc[date][tmp_code[j]] = 1
            if count == tmp:
                break
    signal = signal.ffill()
    #     print(signal.dropna(axis=1,how='all'))
    return signal


def strategy2_GB(data: backtestData2):

    df = pd.read_csv(os.getcwd() + '\\MR result data\\' + "result_GB.csv", index_col=1)
    pred = df.loc[:, ['stcode', 'prob']]
    prices = pd.read_csv(os.getcwd() + '\\MR result data\\' + "price.csv", index_col=0)
    signal = pd.read_excel(os.getcwd() + '\\BT raw data\\' + "time.xlsx", index_col=0)
    for code in prices.columns.values:
        signal[int(code)] = np.nan
    epsilon = 0.05
    buy = 0.7
    sell = 0.5
    max = 5

    hold = []
    for i in range(signal.index.shape[0]):
        out = []
        date = prices.index[i]
        j = 0
        while j < len(hold):
            ran = random.random()
            if ran < epsilon and len(hold) != 0:
                hold.pop(random.randint(0, len(hold) - 1))
            try:
                tmp = pred.loc[date]
                tmp = tmp[tmp['stcode'] == hold[j]]['prob'].values[0]
                if tmp < 1 - sell:
                    signal.loc[date][hold[j]] = 0
                    z = hold.pop(j)
                    out.append(z)
                    continue
                else:
                    j += 1
            except:
                j += 1
        tmp = max - len(hold)
        tmp_pred = pred.loc[date].tail(tmp + max)['prob'].values
        tmp_code = pred.loc[date].tail(tmp + max)['stcode'].values
        count = 0
        for j in range(len(tmp_pred)):
            if float(tmp_pred[j]) > buy and tmp_code[j] not in hold and tmp > 0 and tmp_code[j] not in out:
                hold.append(tmp_code[j])
                count += 1
                if tmp_code[j] in signal:
                    signal.loc[date][tmp_code[j]] = 1
                else:
                    signal[tmp_code[j]] = np.nan
                    signal.loc[date][tmp_code[j]] = 1
            if count == tmp:
                break
    signal = signal.ffill()
    #     print(signal.dropna(axis=1,how='all'))
    return signal


def strategy1_NB(data: backtestData2):

    df = pd.read_csv(os.getcwd() + '\\MR result data\\' + "result_NB.csv", index_col=1)
    pred = df.loc[:, ['stcode', 'prob']]
    prices = pd.read_csv(os.getcwd() + '\\MR result data\\' + "price.csv", index_col=0)
    signal = pd.read_excel(os.getcwd() + '\\BT raw data\\' + "time.xlsx", index_col=0)
    for code in prices.columns.values:
        signal[int(code)] = np.nan
    buy = 0.7
    sell = 0.5
    max = 5

    hold = []
    for i in range(signal.index.shape[0]):
        out = []
        date = prices.index[i]
        j = 0
        while j < len(hold):
            try:
                tmp = pred.loc[date]
                tmp = tmp[tmp['stcode'] == hold[j]]['prob'].values[0]
                if tmp < 1 - sell:
                    signal.loc[date][hold[j]] = 0
                    z = hold.pop(j)
                    out.append(z)
                    continue
                else:
                    j += 1
            except:
                j += 1
        tmp = max - len(hold)
        tmp_pred = pred.loc[date].tail(tmp + max)['prob'].values
        tmp_code = pred.loc[date].tail(tmp + max)['stcode'].values
        count = 0
        for j in range(len(tmp_pred)):
            if float(tmp_pred[j]) > buy and tmp_code[j] not in hold and tmp > 0 and tmp_code[j] not in out:
                hold.append(tmp_code[j])
                count += 1
                if tmp_code[j] in signal:
                    signal.loc[date][tmp_code[j]] = 1
                else:
                    signal[tmp_code[j]] = np.nan
                    signal.loc[date][tmp_code[j]] = 1
            if count == tmp:
                break
    signal = signal.ffill()
    #     print(signal.dropna(axis=1,how='all'))
    return signal


def strategy2_NB(data: backtestData2):

    df = pd.read_csv(os.getcwd() + '\\MR result data\\' + "result_NB.csv", index_col=1)
    pred = df.loc[:, ['stcode', 'prob']]
    prices = pd.read_csv(os.getcwd() + '\\MR result data\\' + "price.csv",index_col=0)
    signal = pd.read_excel(os.getcwd() + '\\BT raw data\\' + "time.xlsx", index_col=0)
    for code in prices.columns.values:
        signal[int(code)] = np.nan
    epsilon = 0.05
    buy = 0.7
    sell = 0.5
    max = 5

    hold = []
    for i in range(signal.index.shape[0]):
        out = []
        date = prices.index[i]
        j = 0
        while j < len(hold):
            ran = random.random()
            if ran < epsilon and len(hold) != 0:
                hold.pop(random.randint(0, len(hold) - 1))
            try:
                tmp = pred.loc[date]
                tmp = tmp[tmp['stcode'] == hold[j]]['prob'].values[0]
                if tmp < 1 - sell:
                    signal.loc[date][hold[j]] = 0
                    z = hold.pop(j)
                    out.append(z)
                    continue
                else:
                    j += 1
            except:
                j += 1
        tmp = max - len(hold)
        tmp_pred = pred.loc[date].tail(tmp + max)['prob'].values
        tmp_code = pred.loc[date].tail(tmp + max)['stcode'].values
        count = 0
        for j in range(len(tmp_pred)):
            if float(tmp_pred[j]) > buy and tmp_code[j] not in hold and tmp > 0 and tmp_code[j] not in out:
                hold.append(tmp_code[j])
                count += 1
                if tmp_code[j] in signal:
                    signal.loc[date][tmp_code[j]] = 1
                else:
                    signal[tmp_code[j]] = np.nan
                    signal.loc[date][tmp_code[j]] = 1
            if count == tmp:
                break
    signal = signal.ffill()
    #     print(signal.dropna(axis=1,how='all'))
    return signal


# set up parameters
start_time = None
end_time = None

timelist1 = pd.read_csv(os.getcwd() + '\\MR result data\\' + 'price.csv')['trade_dt'].to_list()
# print(timelist1)

df0 = pd.read_csv(os.getcwd() + '\\MR result data\\' + 'price.csv', index_col=0)
data1 = backtestData2({'close': df0})

k = 5  # choose the top 5 industry


def backtest_strategy(strategy):
    # backtesting
    st = time()
    parameters = {
        'data': data1,
        'strategy': strategy,
        'starttime': start_time,
        'endtime': end_time,
        'trade_at': 'close',
        'trade_fee': 0.00,
        'rf': 0.00
    }

    model = backtest2(**parameters)
    model.run(plot=True, mode='real')

    print(model.calc_stats())

    print('time used:', time() - st)

    a = model.excess_return
    # print(a)


for i in [strategy_LR, strategy_NB, strategy_GB, strategy1_LR, strategy2_LR, strategy1_NB,
          strategy2_NB, strategy1_GB, strategy2_GB]:
    backtest_strategy(strategy=i)