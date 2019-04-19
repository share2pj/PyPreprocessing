"""
- グルーピング
- 合計値
- カウント
- 最大・最小・平均・中央・パーセンタイル
- 分散値と標準偏差値
- 最頻値
- ランキング
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from seaborn import load_dataset

boston = load_boston()
df = pd.DataFrame(data=np.c_[boston['data'], boston['target']],
                  columns=np.append(boston['feature_names'], 'price'))

mpg = load_dataset('mpg')
"-------------------------------------------------------"
# 集約処理 - 複数列を用いたグルーピング、合計の計算

# 集約単位をAGEとRMの組み合わせを指定
# 集約したデータからpriceを取り出し、sum関数に適用することで売上合計金額を算出
result = df \
    .groupby(['AGE', 'RM'])['price'] \
    .sum().reset_index()

# 売上合計金額の列名がpriceになっているので、price_sumに変更
result.rename(columns={'price': 'price_sum'}, inplace=True)
print(result)
"-------------------------------------------------------"
# agg関数を利用して、集約処理をまとめて指定
"-------------------------------------------------------"
# 集約処理 - 出現回数のカウント

#  - originを対象にcount関数を適用
#  - cnameを対象にnunique関数を適用
result = mpg \
  .groupby('cylinders') \
  .agg({'origin': 'count', 'name': 'nunique'})

# reset_index関数によって、列番号を振り直す（inplace=Trueなので、直接resultを更新）
result.reset_index(inplace=True)
result.columns = ['cylinders', 'origin_cnt', 'type_cnt']
print(result)
"-------------------------------------------------------"
# 集約処理 - 最大・最小・平均・中央・パーセンタイル

# priceを対象にmax/min/mean/median関数を適用
# Pythonのラムダ式をagg関数の集約処理に指定
# ラムダ式にはnumpy.percentileを指定しパーセントタイル値を算出（パーセントは20指定）
result = df \
  .groupby('RM') \
  .agg({'price': ['max', 'min', 'mean', 'median',
                  lambda x: np.percentile(x, q=20)]}) \
  .reset_index()
result.columns = ['RM', 'price_max', 'price_min', 'price_mean',
                  'price_median', 'price_20per']
print(result)
"-------------------------------------------------------"
# 集約処理 - 分散値と標準偏差値

# priceに対して、var関数とstd関数を適用し、分散値と標準偏差値を算出
result = df \
  .groupby('RM') \
  .agg({'price': ['var', 'std']}).reset_index()
result.columns = ['RM', 'price_var', 'price_std']

# データ数が1件だったときは、分散値と標準偏差値がnaになっているので、0に置き換え
result.fillna(0, inplace=True)
print(result)
"-------------------------------------------------------"
"-------------------------------------------------------"
# 集約処理 - 最頻値

# round関数で四捨五入した後に、mode関数で最頻値を算出
print(df['price'].round(-3).mode())
"-------------------------------------------------------"
# 集約処理 - ランキング
# rank関数
# method: 同率の値が複数存在する場合の処理 defaultはaverage
#  - max/min/average 3位以降に重複が3件ある場合　4位タイと呼ぶか、6位と呼ぶか5位と呼ぶか
"-------------------------------------------------------"
# model_noを新たな列として追加
# ascending:True = 昇順
mpg['model_no'] = mpg \
  .groupby('origin')['cylinders'] \
  .rank(ascending=True, method='first')

print(mpg)
"-------------------------------------------------------"
# 製造国ごとの車種数（origin_cnt_tb）を計算
origin_cnt_tb = mpg.groupby('origin').size().reset_index()
origin_cnt_tb.columns = ['origin', 'origin_cnt']

# 車種数をもとに順位を計算
# ascending:False = 降順
origin_cnt_tb['origin_cnt_rank'] = origin_cnt_tb['origin_cnt'] \
  .rank(ascending=False, method='min')

print(origin_cnt_tb.sort_values(by=['origin_cnt_rank'], ascending=True))
"-------------------------------------------------------"
