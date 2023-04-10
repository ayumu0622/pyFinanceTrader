import os
import gzip
from datetime import datetime, timedelta, date
import pandas as pd
import time
from time import sleep
from urllib import request
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas_ta as ta
import copy
import sys
import math
import pickle
from IPython.display import Image
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_1samp
import japanize_matplotlib
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import optuna
from typing import NamedTuple, Optional
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

def calc_features(df):
    """
    テクニカル指標に基づく特徴量を作成する関数
    
    Parameters
    ==========
    df: pandas dataframe
        各時刻のohlcvが格納されたdataframe
    
    Returns
    =======
    df_: pandas dataframe
        特徴量が追加されたdataframe
    """
    
    df_ = df.copy()
    
    # pandas_taの入力は必ずpandas series
    close = df_['close']
    open = df_['open']
    high = df_['high']
    low = df_['low']
    volume = df_['volume']
    
    ichimoku_visible, _ = ta.ichimoku(high, low, close)

    df_["ISA_9"] = ichimoku_visible["ISA_9"] /close # spanA
    df_["ISB_26"] = ichimoku_visible["ISB_26"] /close # spanB
    df_["ITS_9"] = ichimoku_visible["ITS_9"] /close # 転換線
    df_["IKS_26"] = ichimoku_visible["IKS_26"] /close # 基準線

    # bbands = ta.bbands(close, length=10)
    # df["BBL_10_2.0"] = bbands["BBL_10_2.0"]  /close # 下の線
    # df["BBU_10_2.0"] = bbands["BBU_10_2.0"] /close # 上の線

    df_["MIDPOINT"] = ta.midpoint(close, timeperiod=16) / close
    df_["T3"] = ta.t3(close, timeperiod=4, vfactor=0) / close
    df_["TEMA"] = ta.tema(close, timeperiod=24) / close
    df_["TRIMA"] = ta.trima(close, timeperiod=24) / close

    df_["MFI"] = ta.mfi(high, low, close, volume, timeperiod=16)

    df_["APO"] = ta.apo(close, fastperiod=12, slowperiod=26, matype=0)

    df_["WILLR"] = ta.willr(high, low, close, timeperiod=16)

    df_["NATR"] = ta.natr(high, low, close, timeperiod=1)
   
    
    #単純移動平均
    df_['SMA5'] = ta.sma(close, length=5) / close
    df_['SMA10'] = ta.sma(close,length=10) / close
    df_['SMA25'] = ta.sma(close,length=25) / close

    #加重移動平均
    df_['WMA5'] = ta.wma(close,length=5) / close
    df_['WMA10'] = ta.wma(close,length=10) / close
    df_['WMA25'] = ta.wma(close,length=25) / close

    #指数加重移動平均
    df_['EMA5'] = ta.ema(close,length=5) / close
    df_['EMA10'] = ta.ema(close,length=10) / close
    df_['EMA25'] = ta.ema(close,length=25) / close

    #モメンタム
    df_['MOM7'] = ta.mom(close,length=7) / close

    #ROC(変化率)
    df_['ROC10'] = ta.roc(close,length=10)

    # 回帰
    df_['REG'] = ta.linreg(close,length=5) / close

    df_['ATR'] = ta.atr(df_['high'], df_['low'], close, length=14) / close

    #RSI(Relative Strength Index)
    #買われ過ぎ，売られ過ぎを数値で把握しようとするもの
    df_['RSI5'] = ta.rsi(close,length=5)
    df_['RSI10'] = ta.rsi(close,length=10)

    #ストキャスティック
    #RSIと同様買われすぎ,売られ過ぎを把握する
    stoch = ta.stoch(df_['high'], df_['low'], close, k=14, d=3)
    df_ = pd.concat([df_, stoch], axis=1)
    
    #Bollinger Bands
    bbands1 = ta.bbands(close, length=25, std=1)
    bbands2 = ta.bbands(close, length=25, std=2)
    bbands3 = ta.bbands(close, length=25, std=3)
    
    for bbands in [bbands1, bbands2, bbands3]:
        for i, col in enumerate(bbands.columns):
            # i = 3, i = 4に対応するのはBBB, BBP
            # この二つは無次元量なのでcloseで割らない
            if i < 3:
                bbands[col] /= close

    df_ = pd.concat([df_, bbands1, bbands2, bbands3], axis=1)

    #MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    
    for col in macd.columns:
        macd[col] /= close
    
    df_ = pd.concat([df_, macd], axis=1)
    
    df_ = df_.dropna()
    
    return df_


def make_first_prediction(df):
    """
    1次モデルの予測を作成する関数
    
    Parameters
    ==========
    df : pandas dataframe
        各時刻の価格情報が格納されたdataframe
    
    Returns
    =======
    df_: pandas dataframe
        1次モデルの予測が'side'カラムに格納されたdataframe
    """
    
    df_ = df.copy()

    bbands = ta.bbands(df_['close'], length=25, std=2) # ボリンジャーバンド
    sma5 = ta.sma(df_['close'], length=5)  #単純移動平均
    sma25 = ta.sma(df_['close'], length=25)

    # ボリンジャーバンドによる予測を導出
    df_["bb_sell"] = np.where(bbands["BBL_25_2.0"] > df_["high"], -1, 0)
    df_["bb_buy"] = np.where(bbands["BBU_25_2.0"] < df_["low"], 1, 0)
    df_["bb_side"] = df_.apply(lambda row: row["bb_buy"] + row["bb_sell"], axis=1)

    # ゴールデン, デッドクロスによる予測を導出
    diff = sma5 - sma25
    df_['sma_side'] = np.where((np.sign(diff) - np.sign(diff.shift(1)) == 2), 1, 0) # ゴールデンクロス
    df_['sma_side'] = np.where((np.sign(diff) - np.sign(diff.shift(1)) == -2), -1, df_['sma_side']) # デッドクロス 
    
  

    #MACDがデットクロスした時にRSIが80%以上である(sell)
    df_['RSI10_80'] = 0
    df_.loc[(df_['RSI10'] >= 80) & (df_['ROC10'] > 0) & ((df_['sma_side'].diff(1) == -1) | (df_['sma_side'].diff(2) == -1) | (df_['sma_side'].diff(3) == -1)), 'RSI10_80'] = 1

    #MACDがゴールデンクロスした時にRSIが30%以下である(buy)。
    df_['RSI10_30'] = 0
    df_.loc[(df_['RSI10'] <= 30) & (df_['ROC10'] < 0) & ((df_['sma_side'].diff(1) == 1) | (df_['sma_side'].diff(2) == 1) | (df_['sma_side'].diff(3) == 1)), 'RSI10_30'] = 1

    df_['sell'] = np.where((((df_['bb_side'] + df_['sma_side']) < 0) | (df_['RSI10_80'] == 1)), -1, 0)
    df_['buy'] = np.where((((df_['bb_side'] + df_['sma_side']) > 0) | (df_['RSI10_30'] == 1)), 1, 0)
    df_['side'] = df_['buy'].values + df_['sell'].values
    
  
    df_['sell'] = np.where((df_['bb_side'] + df_['sma_side'] < 0), -1, 0)
    df_['buy'] = np.where((df_['bb_side'] + df_['sma_side'] > 0), 1, 0)
    df_['side'] = df_['buy'].values + df_['sell'].values


    #使わない特徴量は消しておく
    df_ = df_.drop(['bb_sell', 'bb_buy', 'bb_side', 'sma_side'], axis=1)
    
    return df_


def make_second_labels(df, col):

    h = 16
    df['ret'] = (df['close'].diff(h).shift(-h)[:-h] / df['close'][:-h])
#リターン（0.009, -0.0003, etc） とサイド（−１、０、１）をかけて、予想が当たってたら＋、ベットしてなかったら０、外れてたらマイナスに変換
    df['metalabel'] = 0
    df.loc[(df['ret'] >= 0.005),'metalabel'] = 1
    df.loc[(df['ret'] <= -0.005), 'metalabel'] = -1
    #マイナスは０に変換して、答えのラベルを作る。
    df['metalabel'] = np.sign(df['metalabel'] * df['side'])
    df['metalabel'] = df['metalabel'].apply(lambda x: max(0, x))

    return df


def purged_cv(df, n_split=5, n_purge=5):
    """
    Purged CVによる検証を行うために，データを分割する関数
    
    Parameters
    ==========
    df: pandas DataFrame
        特徴量と目的変数を格納したDataFrame
    n_split: int
        CVの分割数
    n_purge: int
        パージングするレコード数
    
    Returns
    =======
    indices: list
        各要素が，各foldの[学習期間のインデックス, 検証期間のインデックス]になっているリスト
    """
    
    length = len(df)
    idx = list(range(length))
    indices = []
    split_idx = [int(length * i / n_split) for i in range(n_split + 1)]
    
    for i in range(n_split):
        val_from = split_idx[i]
        val_to = split_idx[i + 1]
        val_idx = idx[val_from : val_to]
        train_idx = []
        if val_from - n_purge > 1:
            train_idx += idx[:val_from - n_purge]
        if val_to + n_purge < len(df):
            train_idx += idx[val_to + n_purge :]
        indices.append([train_idx, val_idx])
    
    return indices

def training(df, features, y, model, n_split=5, n_purge=10):
    """
    Purged CVによりモデルを学習させ，各学習済みモデルによる予測を作成する関数
    
    Parameters
    ==========
    df: pandas dataframe
        特徴量とラベルが格納されたdataframe
    features: list
        モデルの学習に用いる特徴量の名前(str)が格納されたlist
    y: str
        モデルの学習に用いるラベルの名前
    model: 
        学習に用いるモデル
    n_split: int
        cpcvの分割数
    n_purge: int
        パージングするレコード数
    
    Returns
    ==========
    ret: pandas dataframe
        二次モデルの予測が格納されたdataframe
    """
    n_split = n_split

    cv = purged_cv(df, n_split=n_split, n_purge=n_purge)

    # 一次モデルの出力を二次モデルの学習に用いる
    features_ = copy.deepcopy(features)
    #features_.append('side')

    # 結果格納用のDataFrame
    ret = df[[y]].copy()
    ret['probability'] = [0] * len(ret)
    for train_idx, val_idx in cv:
        train_idx = df.index[train_idx]
        val_idx = df.index[val_idx]
        model.fit(df.loc[train_idx, features_], df.loc[train_idx, y])
        ret.loc[val_idx, 'probability'] = model.predict_proba(df.loc[val_idx, features_])[:, 1]

    return ret

def run(trial, df, features, model = 'lgb'):
    """
    モデルのハイパーパラメータの最適化を行う関数
    
    Parameters
    ==========
    trial: optuna.trial.Trial

    Returns
    =======
    criterion: float
        各試行で得られたクロスエントロピーの値
    """
    
    if model == 'logi':
        rfc = LogisticRegression(random_state = 42)

    ##add model
        

    Xy = df.dropna(axis=0)
    ret = training(Xy, features, 'metalabel', model=rfc, n_split=10, n_purge=10)
    Xy = pd.concat([Xy, ret], axis=1)
    criterion = log_loss(ret['metalabel'].values, ret['probability'].values)
    
    return criterion

def preprocess(df):
    """
    前処理をまとめて行う関数

    Parameters
    ==========
    df: pandas dataframe
        ohlcデータが格納されたdataframe
    
    Returns
    ==========
    train_df: pandas dataframe
        特徴量が格納されたdataframe
    """

    train_df = calc_features(df)
    
    return train_df

def make_models(df, features, best_params, y):
    """
    与えられたベストパラメータを用いてPurged CVにより学習済みモデルを作成する関数
    
    Parameters
    ==========
    df: pandas dataframe
        特徴量とラベルが格納されたdataframe
    features: list
        モデルの学習に用いる特徴量の名前(str)が格納されたlist
    y: str
        モデルの学習に用いるラベルの名前
    best_params: dictionary
        探索で得たモデルのベストパラメータ
    
    Return
    ======
    models: list
        学習済みモデルを格納したリスト
    """
    models = []
    train_idx = df.index

    model = LogisticRegression(**best_params['logi'])
    model.fit(df.loc[:, features], df.loc[:, y])
    models.append(model)

    return models

def get_model(raw_df, train_df):
    """
    与えられた日付の範囲のデータを加工し，学習済みモデルを生成する関数
    
    Parameters
    ==========
    raw_df: pandas dataframe
        preprocessが施されていない訓練データ
    train_df: pandas dataframe
        preprocessが施された訓練データ
    
    Returns
    =======
    models: list
        CVにより分割された各foldのデータを用いて作成された学習済みモデルが格納されたリスト
    """

    features = train_df.columns.to_list()

    print('making first predictions')
    train_df = make_first_prediction(train_df)

    # 一次モデルの出力を二次モデルの予測に用いる
    features.append('side')

    print('making second labels')
    train_df = make_second_labels(train_df, 'side')
    
    best_params = {}
    print('seaching best parameters')
    
    study3 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study3.optimize(lambda trial: run(trial, train_df, features, 'logi'), n_trials=5, show_progress_bar=True)
    best_param3 = study3.best_params
    best_params['logi'] = best_param3

    print(f'best_params: {best_params}')
    
    print('creating models')
    models = make_models(train_df, features, best_params, y='metalabel')
    
    print('done')
    
    return models

def make_second_prediction(df, features, models):
    """
    加工済みデータと学習済みモデルを受け取り，2次モデルの出力を返す関数
    
    Parameters
    ==========
    df: pandas dataframe
        特徴量と1次モデルの予測が格納されたdataframe
    features: list
        モデルの推論に用いる特徴量の名前(str)が格納されたlist
    models: list
        学習済みモデル， 学習に用いたデータの範囲の情報が格納されたリスト
    
    Returns
    =======
    df_: pandas dataframe
        2次モデルの予測が'probability'カラムに格納されたdataframe
    """
    
    df_ = df.copy()

    model3 = models[0]
   
    df_['probability'] = model3.predict_proba(df_.loc[:, features])[:, 1] 


    
    
    return df_


def get_signal(preprocessed_df: pd.DataFrame, models: list=None):
    """
    preprocessed_dfを使ってシグナルを作成する関数

    Parameters
    ==========
    preprocessed_df: pd.DataFrame
        preprocess関数により前処理が施されたデータフレーム
    models: list
        get_modelの返り値. (optional)

    Returns
    =======
    df_: pd.DataFrame
        signal情報を持つpd.DataFrame
    """
    df_ = preprocessed_df.copy()
    features = df_.columns.to_list()
    
    df_ = make_first_prediction(df_)
    features.append('side') # 一次モデルの出力を二次モデルの学習に用いる
    
    if models:
        print('making second predictions')
        df_ = make_second_prediction(df_, features, models)
    
        df_['pred_buy'] = df_.apply(lambda row: row['buy'] if row['probability'] > 0 else 0, axis=1)
        df_['pred_sell'] = df_.apply(lambda row: row['sell'] if row['probability'] > 0 else 0, axis=1)
        df_['side'] = df_['pred_buy'] + df_['pred_sell']  

        df_['size'] = abs(df_['side'].values)  
        df_['size'] = df_['probability'].apply(lambda x : 2 * norm.cdf((x - 0)/(x * (1.0 - x)) ** 0.5)-1) 
        df_.loc[(0.3 >= df_['probability']), 'size'] = 0 

    else:
        df_['side'] = df_['buy'] + df_['sell']  
        df_['size'] = abs(df_['side'].values)  
        df_['size'] = df_['probability'].apply(lambda x : 2 * norm.cdf((x - 0)/(x * (1.0 - x)) ** 0.5)-1) 
        df_.loc[(0.3 >= df_['probability']), 'size'] = 0
    
    df_['size'] = abs(df_['side'].values)  
    df_['size'] = df_['probability'].apply(lambda x : 2 * norm.cdf((x - 0)/(x * (1.0 - x)) ** 0.5)-1) 
    df_.loc[(0.3 >= df_['probability']), 'size'] = 0  

    df_['side'] = df_['side'].apply(lambda x : "BUY" if x == 1 else x)
    df_['side'] = df_['side'].apply(lambda x : "SELL" if x == -1 else x)
    df_['order_flag'] = df_['size']
    

    return df_






class Order(NamedTuple):
    """
    注文情報を持つクラス
    """
    type: str
    side: str
    size: float
    price: Optional[float]

def get_order(current_time, asset_info, signals=None):
    """
    注文時刻，その時刻におけるポジションの状況，OHLCVから得たシグナルを元に注文を作成する関数
    
    Parameters
    ==========
    current_time: pandas.Timestamp
        注文を出す時刻
    asset_info: dict
        注文時におけるポジションの情報
        pos_size: ポジションサイズ(単位はBTC, sideがLONGならば正, SHORTならば負)
        pos_avgp: 平均entry価格
        pos_side: ボジションサイド(LONG, SHORT, ZERO)
        totalpos_abs: 現在の保有ポジション(size×priceの絶対値)
        max_limit_totalpos_abs: 現在のポジションから算出される最大保有ポジション
                                (現在の保有ポジションの方向で積み上げた際の上限値で、絶対値で表現)
        margin: 現在保有の証拠金額
        min_limit_margin: 現在の保有ポジションから算出される最小必要証拠金額
    signals: pandas.DataFrame
        注文作成に必要な情報が格納されたデータフレーム



    Returns
    =======
    order_lst: list (中身はOrderクラス)
        current_timeにおける注文情報が格納されている
        'type','side','size','price'の４項目
    """

    df_current = signals.loc[current_time, :]
    order_lst = []
    pos_size = asset_info['pos_size']
    pos_side = asset_info['pos_side']
    pos_avgp = asset_info['pos_avgp']

    #max_position check
    max_pos = 5
    max_pos_flag = abs(pos_size) > max_pos
    # profit-take and loss-cut check
    pt_lc_thre = 0.005
    pt_flag = df_current['close'] > pos_avgp * (1 + pt_lc_thre)
    lc_flag = df_current['close'] < pos_avgp * (1 - pt_lc_thre)

    if (pos_side!='ZERO') and (max_pos_flag or pt_flag or lc_flag):
        if pos_side == 'LONG':
            order_lst.append(Order('MARKET', 'SELL', abs(pos_size), None))
        elif pos_side == 'SHORT':
            order_lst.append(Order('MARKET', 'BUY', abs(pos_size), None))
    
    else:
        if df_current['order_flag'] == 1:
            order_lst.append(Order('MARKET', df_current['side'], df_current['size'], None))

    return order_lst
