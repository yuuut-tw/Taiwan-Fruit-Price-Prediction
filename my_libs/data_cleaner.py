import pandas as pd
import numpy as np
import re
import os
from functools import reduce

def replacer(data):
    return re.sub(",|-", "", data).replace("", np.nan).fillna(method="ffill")


def df_cleaner(df):
    # step 1 讀取資料(選取需求欄位)
    cols = ['日期', '市場', '平均價(元/公斤)', '交易量(公斤)']
    df = df[cols]
    
    # step 2 清洗與補缺值
    ## 時間 -> 日期: 民國->西元年
    df["日期"] = df["日期"].apply(lambda x: re.sub("\d{3}", "{}".format((int(x.split("/")[0]) + 1911)), x))
    df["日期"] = pd.to_datetime(df["日期"])
    
    # step 3 去除成交量中的雜質(數據型態為object都處理)
    target_cols = list(df.select_dtypes("object"))
    df[target_cols] = df[target_cols].apply(lambda x: x.str.strip(" "))
    
    # step 4 更改市場名稱
    df["市場"] = df["市場"].apply(lambda x:x.split(" ")[1])
    
    # step 5 找出缺失值(re.sub) & 轉換型態
    try:
        df["平均價(元/公斤)"] = df["平均價(元/公斤)"].apply(lambda x:re.sub(",|-", "", x)).replace("", np.nan).fillna(method="ffill").astype("float")
        df["交易量(公斤)"] = df["交易量(公斤)"].apply(lambda x:re.sub(",|-", "", x)).replace("", np.nan).fillna(method="ffill").astype("float")
    except:
        pass
    
    return df



def df_merger(df, df_same, df_sub, fruit, market):
    
    product, same_type, substitution = fruit[0], fruit[1], fruit[2]
    
    # step 1 找出特定市場
    df_market =  df.loc[df["市場"] == market]
    same = df_same.loc[df_same["市場"] == market]
    sub = df_sub.loc[df_sub["市場"] == market]
    
    # step 2 進行merge，以df_market的日期為鍵進行left join
    dfs = [df_market, same, sub]
    df_merged = reduce(lambda left,right: pd.merge(left, right, on="日期", how="left"), dfs)

    #print(f"{market} 原始資料筆數(合併後): {df_merged.shape[0]}")
    
    # stpe 3 將日期設為index
    df_merged.set_index("日期", inplace=True)

    # stpe 4 篩選出必要欄位&重新命名
    cols = list(df_merged.columns) 
    df_merged = df_merged[cols[1:3] + [cols[4]] + [cols[7]]]
    df_merged.columns = [f"{product}_平均價", f"{product}_交易量", f"{same_type}_平均價", f"{substitution}_平均價"]

    # step 5 resample補齊每日資料(補值: 插值 -> 前值 -> 後值)
    df_merged = df_merged.resample("D").interpolate().fillna(method="ffill").fillna(method="bfill").applymap(lambda x: round(x,1))

    # step 6 新增欄位(前日價格 & 5日移動平均)
    df_merged[f"{product}_前日平均價"] = df_merged[f"{product}_平均價"].shift(1).fillna(method="bfill")
    df_merged[f"{product}_5日平均價"] = round(df_merged[f"{product}_平均價"].rolling(5).mean().fillna(method="bfill"), 1)
    
    # step 7 將日期從index還原為column
    df_merged.reset_index(inplace=True)
    
    return df_merged