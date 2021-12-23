# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from types import FunctionType as function
from pylab import mpl
from datetime import datetime
import os


# =============================================================================
# backtestData2: 储存数据的类
# =============================================================================
class backtestData2():
    """储存格式已经对齐的回测数据。"""

    def __init__(self, data_dict: dict, is_matched: bool = False):
        """
        Parameters
        ----------
        data_dict : dict{str: pd.DataFrame}
            数据集.
        is_matched : bool, optional
            数据是否已经对齐. The default is False.
        """
        self.template = None

        if is_matched:
            self.dt = data_dict
            return

        self.dt = {}
        for i in data_dict:
            self.dt[i] = self._match_data(data_dict[i])

    def add(self, new: dict):
        """新增数据"""
        for i in new:
            self.dt[i] = self._match_data(new[i])

    def select(self, tradetime: pd.DatetimeIndex):
        """选择指定时间范围的数据"""
        selected_data = {i: self.dt[i].loc[tradetime] for i in self.dt}
        return backtestData2(selected_data, is_matched=True)

    def _match_data(self, Input: pd.DataFrame):
        """对齐数据"""
        check_type(Input, pd.DataFrame)

        if self.template is None:
            self.template = Input
            return Input

        matched_data = self.template.copy() + np.nan
        for col in matched_data:
            matched_data[col] = Input[col]
        return matched_data


# =============================================================================
# stats: 统计回测结果
# =============================================================================
class stats():
    """计算统计量"""

    def __init__(self, nv_curve, rf, signal):
        self.nv_curve = nv_curve
        self.rf = rf
        self.signal = signal
        self.trading_days_year = 250

    def maximum_drawdown(self):
        dd_end = (self.nv_curve / np.maximum.accumulate(self.nv_curve)).idxmin()
        # np.maximum.accumulate: 当日之前历史最高价值的序列, idxmin(): 序列最小值的索引
        dd_start = (self.nv_curve[:dd_end]).idxmax()  # idxmax(): 序列最大值的索引
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
        # 年化波动率=收益率标准差*(n^0.5), n: 按日=250, 周=52, 月=12, 年=1
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


# =============================================================================
# backtest2: 回测框架
# =============================================================================
class backtest2():
    """回测框架: 提供回测功能"""

    def __init__(self, *args, **kwargs):
        self.set_params(*args, **kwargs)

    def set_params(self,
                   data: backtestData2,
                   strategy: function = None,
                   starttime: (str, datetime) = None,
                   endtime: (str, datetime) = None,
                   trade_at: str = None,
                   trade_asset: list = None,
                   trade_fee: float = None,
                   rf: (int, float) = None
                   ):
        """
        回测参数

        Parameters
        ----------
        data : backtestData2, optional
            数据. The default is None.
        strategy : function, optional
            策略函数(backtestData2)->signal. The default is None.
        starttime : (str, datetime), optional
            回测开始时间. The default is None.
        endtime : (str, datetime), optional
            回测结束时间. The default is None.
        trade_at : str, optional
            预设成交价格. The default is None.
        trade_asset : list(str), optional
            交易标的. The default is None.
        trade_fee: float, optional
            交易费用(双边). The default is None.
        rf : (int, float), optional
            Risk-free rate. The default is None.

        """

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
        """运行回测"""
        # check the attributes
        self._check_attr('data')
        self._check_attr('strategy')

        # loading data
        price = self.data.dt[self.trade_at]
        timelist = price.index

        if self.starttime is None:
            self.starttime = timelist[0]  # 默认从第一天开始
        if self.endtime is None:
            self.endtime = timelist[-1]  # 默认在最后一天结束
        self.tradetime = timelist[(timelist >= self.starttime) &
                                  (timelist <= self.endtime)]  # 实际回测时间段

        self.trade_price = price.loc[self.tradetime]
        self.stocknames = self.trade_price.columns
        self.retn = (self.trade_price / self.trade_price.shift() - 1).fillna(0)
        self.valid = ~np.isnan(self.trade_price)  # 和trade_price结构相同，Nan-->False, ~Nan-->True
        if self.trade_asset is not None:
            is_trade_asset = self.valid * 0
            is_trade_asset[self.trade_asset] = 1
            self.valid = self.valid & is_trade_asset

        # 生成信号
        data_used = self.data.select(self.tradetime)  # 和trade_price区别：
        # data_used: dict, 包含data.dt中所有k, v; trade_price: df, 从data.dt中取出k=trade_at的v(df)
        self.signal = self.strategy(data_used)
        self.norm_signal = self._signal_normalize(self.signal)

        # trade according to the signals
        self.trade_simulator(mode=mode)

        # 计算excess return
        self.baseline = (1 + self.retn.mean(axis=1)).cumprod()  # baseline
        self.excess_return = self.nv_curve / self.baseline  # excess return

        if plot: self.plot()

        return self

    def trade_simulator(self, mode: str = 'real'):
        """交易模拟"""

        if mode == 'quick':  # 粗略地计算交易费用
            fee_rate = self.trade_fee / 2  # 交易费率
            retn = self.retn.values  # 单股收益率
            norm_signal = self.norm_signal.values  # 标准化信号

            port_return = (retn * shift(norm_signal, fill_value=0)).sum(axis=1)  # 资产收益率
            position_before = div((shift(norm_signal, fill_value=0) * (1 + retn)), 1 + port_return)  # 调仓前仓位
            position_change = norm_signal - position_before  # 仓位变化
            fee = np.abs(position_change).sum(axis=1) * fee_rate  # 交易费用
            nv_curve = ((1 + port_return) * (1 - fee)).cumprod()  # 净值曲线
            real_position = mul(norm_signal, nv_curve)
            cash = nv_curve * (1 - norm_signal.sum(axis=1))  # 现金
            nv_shift = shift(nv_curve, fill_value=1)
            accumulated_fee = np.cumsum(nv_shift * (1 + port_return) * fee)  # 累积交易费用

            self.real_position = self._np2df(real_position)
            self.cash = self._np2sr(cash)
            self.accumulated_fee = self._np2sr(accumulated_fee)  # 累积交易费用
            self.nv_curve = self._np2sr(nv_curve)

        elif mode == 'real':  # 逐日计算交易费用
            fee_rate = self.trade_fee / 2  # 交易费率
            # 这边全都用np做，方便loc
            retn = self.retn.values  # df-->np, [[第一行的retn], [第二行的retn], ...]
            # print('retn', retn)
            valid = self.valid.values
            norm_signal = self.norm_signal.values  # 标准化信号

            port = np.zeros(self.trade_price.shape[0])  # 资产, shape[0]: df行数(即日期数?)
            position = norm_signal * 0  # 仓位
            cash = port.copy()  # 现金
            fee = port.copy()  # 交易费用

            for i in range(port.shape[0]):
                if i == 0:
                    cash_before = 1  # 初始资金1
                    position_before = position[0] * 0  # 初始仓位0
                    # print(position_before)
                else:
                    cash_before = cash[i - 1]  # 调仓前现金
                    position_before = (1 + retn[i]) * position[i - 1]  # 调仓前仓位, position: 仓位, position_before: 昨天仓位*今天收益
                    # print(position_before)
                port_before = cash_before + position_before.sum()  # 调仓前资产, =调仓前现金+调仓前仓位
                position_change = (port_before * norm_signal[i] - position_before) * valid[i]  # 仓位变化
                n = (position_change < 0) * position_change  # 仓位变化(卖出部分)
                p = (position_change > 0) * position_change  # 仓位变化(买入部分)
                fee[i] = np.abs(position_change).sum() * fee_rate  # 手续费
                cash[i] = cash_before - (fee[i] + position_change.sum())  # 调仓后现金, =调仓前现金-手续费-仓位变化
                if cash[i] < 0:  # 保证调仓后现金不小于0
                    adj_ratio = (cash_before - n.sum()) / (fee[i] + p.sum())
                    position_change = p * adj_ratio + n
                    fee[i] = np.abs(position_change).sum() * fee_rate
                    cash[i] = 0
                position[i] = position_before + position_change  # 调整后仓位, =调仓前仓位+仓位变化
                port[i] = port_before - fee[i]  # 调仓后剩余资产, =调仓前资产-手续费
                # 粗略：port = cash[before] + position[before] - fee
            # print(fee)
            # print(cash)
            # print(position)
            # print('port=', port)
            # print('cash=', cash)

            self.real_position = self._np2df(position)
            self.cash = self._np2sr(cash)
            self.accumulated_fee = self._np2sr(np.cumsum(fee))  # 累积交易费用
            self.nv_curve = self._np2sr(port)

        else:
            raise Exception('Not implemented.')

        return self

    def calc_stats(self):
        """统计"""
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
        """绘图"""
        self._check_attr('nv_curve')
        fig, ax1 = plt.subplots(figsize=(18, 8))
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设定字体
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        plt.plot(self.nv_curve, label='Strategy Net Worth')
        plt.plot(self.baseline, label='Baseline')
        plt.plot(self.excess_return, label='Excess Income')
        plt.xticks(rotation=45)
        plt.title('backtest result: ' + self.strategy.__name__)
        plt.legend()
        plt.savefig(os.getcwd() + '\\BT result fig\\' + 'backtest result' + self.strategy.__name__ + '.png')
        plt.show()

    def _signal_normalize(self, signal: pd.DataFrame):
        """标准化数据"""
        x = signal.fillna(0)
        sum_x = np.maximum(1, x.sum(axis=1))
        return x.div(sum_x, axis=0)

    def _check_attr(self, attr: str):
        """检查属性"""
        if (not hasattr(self, attr)) or (getattr(self, attr) is None):
            raise Exception("'" + attr + "' is not found in this "
                            + type(self).__name__ + ' instance.')

    def _np2df(self, x):
        return pd.DataFrame(x, index=self.tradetime, columns=self.stocknames)

    def _np2sr(self, x):
        return pd.Series(x, index=self.tradetime)


# =============================================================================
# 工具函数
# =============================================================================
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
    """检查对象类型"""
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