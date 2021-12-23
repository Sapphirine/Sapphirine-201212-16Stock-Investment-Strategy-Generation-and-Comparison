# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from types import FunctionType as function
from pylab import mpl
from datetime import datetime



class backtestData():


    def __init__(self, data_dict: dict, is_matched: bool = False):

        self.template = None

        if is_matched:
            self.dt = data_dict
            return

        self.dt = {}
        for i in data_dict:
            self.dt[i] = self._match_data(data_dict[i])

    def add(self, new: dict):

        for i in new:
            self.dt[i] = self._match_data(new[i])

    def select(self, tradetime: pd.DatetimeIndex):

        selected_data = {i: self.dt[i].loc[tradetime] for i in self.dt}
        return backtestData(selected_data, is_matched=True)

    def _match_data(self, Input: pd.DataFrame):

        check_type(Input, pd.DataFrame)

        if self.template is None:
            self.template = Input
            return Input

        matched_data = self.template.copy() + np.nan
        for col in matched_data:
            matched_data[col] = Input[col]
        return matched_data



class stats():

    def __init__(self, nv_curve, rf, signal):
        self.nv_curve = nv_curve
        self.rf = rf
        self.signal = signal
        self.trading_days_year = 250

    def maximum_drawdown(self):
        dd_end = (self.nv_curve / np.maximum.accumulate(self.nv_curve)).idxmin()

        dd_start = (self.nv_curve[:dd_end]).idxmax()
        if self.nv_curve[dd_start] == 0:
            return 0
        return self.nv_curve[dd_end] / self.nv_curve[dd_start] - 1

    def annual_return(self):
        num_of_years = len(self.nv_curve) / self.trading_days_year
        annual_return = pow(self.nv_curve[-1], 1 / num_of_years) - 1
        return annual_return

    def volatility(self):
        nv_return = self.nv_curve / self.nv_curve.shift() - 1
        volatility = nv_return.std() * np.sqrt(self.trading_days_year)

        return volatility

    def sharp_ratio(self):
        sharp_ratio = (self.annual_return() - self.rf) / self.volatility()
        return sharp_ratio

    def calmar_ratio(self):
        calmar_ratio = - self.annual_return() / self.maximum_drawdown()
        return calmar_ratio

    def turnover(self):
        num_of_years = len(self.nv_curve) / self.trading_days_year
        turnover = self.signal.diff().abs().sum().mean(axis=0) / num_of_years
        return turnover



class backtest():


    def __init__(self, *args, **kwargs):
        self.set_params(*args, **kwargs)

    def set_params(self,
                   data: backtestData,
                   strategy: function = None,
                   starttime: (str, datetime) = None,
                   endtime: (str, datetime) = None,
                   trade_at: str = None,
                   trade_asset: list = None,
                   trade_fee: float = None,
                   rf: (int, float) = None
                   ):

        # load parameters
        default_types = self.set_params.__annotations__
        for i in default_types:
            para = locals()[i]
            if (not hasattr(self, i)) or (para != None):
                check_type(para, default_types[i], exception=(None,))
                setattr(self, i, para)

        if self.trade_fee is None: self.trade_fee = 0.003
        if self.rf is None: self.rf = 0.00
        if self.trade_at is None: self.trade_at = 'close'
        return self

    def run(self, plot: bool = False, mode: str = 'real'):


        self._check_attr('data')
        self._check_attr('strategy')


        price = self.data.dt[self.trade_at]
        timelist = price.index

        if self.starttime is None:
            self.starttime = timelist[0]
        if self.endtime is None:
            self.endtime = timelist[-1]
        self.tradetime = timelist[(timelist >= self.starttime) &
                                  (timelist <= self.endtime)]

        self.trade_price = price.loc[self.tradetime]
        self.stocknames = self.trade_price.columns
        self.retn = (self.trade_price / self.trade_price.shift() - 1).fillna(0)
        self.valid = ~np.isnan(self.trade_price)
        if self.trade_asset is not None:
            is_trade_asset = self.valid * 0
            is_trade_asset[self.trade_asset] = 1
            self.valid = self.valid & is_trade_asset


        data_used = self.data.select(self.tradetime)

        self.signal = self.strategy(data_used)
        self.norm_signal = self._signal_normalize(self.signal)

        # trade according to the signals
        self.trade_simulator(mode=mode)


        self.baseline = (1 + self.retn.mean(axis=1)).cumprod()  # baseline
        self.excess_return = self.nv_curve / self.baseline  # excess return

        if plot: self.plot()

        return self

    def trade_simulator(self, mode: str = 'real'):


        if mode == 'quick':
            fee_rate = self.trade_fee / 2
            retn = self.retn.values
            norm_signal = self.norm_signal.values

            port_return = (retn * shift(norm_signal, fill_value=0)).sum(axis=1)
            position_before = div((shift(norm_signal, fill_value=0) * (1 + retn)), 1 + port_return)
            position_change = norm_signal - position_before
            fee = np.abs(position_change).sum(axis=1) * fee_rate
            nv_curve = ((1 + port_return) * (1 - fee)).cumprod()
            real_position = mul(norm_signal, nv_curve)
            cash = nv_curve * (1 - norm_signal.sum(axis=1))
            nv_shift = shift(nv_curve, fill_value=1)
            accumulated_fee = np.cumsum(nv_shift * (1 + port_return) * fee)

            self.real_position = self._np2df(real_position)
            self.cash = self._np2sr(cash)
            self.accumulated_fee = self._np2sr(accumulated_fee)
            self.nv_curve = self._np2sr(nv_curve)

        elif mode == 'real':
            fee_rate = self.trade_fee / 2

            retn = self.retn.values
            print('retn', retn)
            valid = self.valid.values
            norm_signal = self.norm_signal.values

            port = np.zeros(self.trade_price.shape[0])
            position = norm_signal * 0
            cash = port.copy()
            fee = port.copy()

            for i in range(port.shape[0]):
                if i == 0:
                    cash_before = 1
                    position_before = position[0] * 0
                    # print(position_before)
                else:
                    cash_before = cash[i - 1]
                    position_before = (1 + retn[i]) * position[i - 1]
                    # print(position_before)
                port_before = cash_before + position_before.sum()
                position_change = (port_before * norm_signal[i] - position_before) * valid[i]
                n = (position_change < 0) * position_change
                p = (position_change > 0) * position_change
                fee[i] = np.abs(position_change).sum() * fee_rate
                cash[i] = cash_before - (fee[i] + position_change.sum())
                if cash[i] < 0:
                    adj_ratio = (cash_before - n.sum()) / (fee[i] + p.sum())
                    position_change = p * adj_ratio + n
                    fee[i] = np.abs(position_change).sum() * fee_rate
                    cash[i] = 0
                position[i] = position_before + position_change
                port[i] = port_before - fee[i]

            # print(cash)
            # print(position)
            print('port=', port)
            print('cash=', cash)

            self.real_position = self._np2df(position)
            self.cash = self._np2sr(cash)
            self.accumulated_fee = self._np2sr(np.cumsum(fee))  # 累积交易费用
            self.nv_curve = self._np2sr(port)

        else:
            raise Exception('Not implemented.')

        return self


        self._check_attr('nv_curve')
        s = stats(self.nv_curve, self.rf, self.norm_signal)
        stats_dict = {'Annual Return': s.annual_return(),
                      'Annual Volatility': s.volatility(),
                      'Sharp Ratio': s.sharp_ratio(),
                      'Maximum Drawdown': s.maximum_drawdown(),
                      'Calmar Ratio': s.calmar_ratio(),
                      'Annual Turnover Rate': s.turnover()
                      }
        self.stats = pd.Series(stats_dict)
        return self.stats

    def plot(self):

        self._check_attr('nv_curve')
        fig, ax1 = plt.subplots(figsize=(18, 8))
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(self.nv_curve, label='Strategy Net Worth')
        plt.plot(self.baseline, label='Baseline')
        plt.plot(self.excess_return, label='Excess Income')
        plt.xticks(rotation=45)
        plt.title('backtest result: ' + self.strategy.__name__)
        plt.legend()
        plt.savefig('backtest result: ' + self.strategy.__name__)
        plt.show()

    def _signal_normalize(self, signal: pd.DataFrame):

        x = signal.fillna(0)
        sum_x = np.maximum(1, x.sum(axis=1))
        return x.div(sum_x, axis=0)

    def _check_attr(self, attr: str):

        if (not hasattr(self, attr)) or (getattr(self, attr) is None):
            raise Exception("'" + attr + "' is not found in this "
                            + type(self).__name__ + ' instance.')

    def _np2df(self, x):
        return pd.DataFrame(x, index=self.tradetime, columns=self.stocknames)

    def _np2sr(self, x):
        return pd.Series(x, index=self.tradetime)



def shift(x: np.ndarray, periods: int = 1, axis: int = 0, fill_value=None):
    if fill_value is None: fill_value = np.nan
    x_shift = np.roll(x, periods, axis=axis)
    if axis == 0:
        if periods >= 0:
            x_shift[:periods] = fill_value
        else:
            x_shift[periods:] = fill_value
    else:
        if periods >= 0:
            x_shift[:, :periods] = fill_value
        else:
            x_shift[:, periods:] = fill_value
    return x_shift


def div(x: np.ndarray, y: np.ndarray):
    return x / y.reshape(-1, 1)


def mul(x: np.ndarray, y: np.ndarray):
    return x * y.reshape(-1, 1)


def check_type(target, Types, exception=None):

    if type(Types) is not tuple:
        Types = (Types,)
    if (type(exception) is tuple) and (target in exception):
        return True
    if type(target) not in Types:
        joined_names = "' or '".join([x.__name__ for x in Types])
        error_info = ("expected '" + joined_names + "', got '" +
                      type(target).__name__ + "' instead.")
        raise TypeError(error_info)
    return True
# %%