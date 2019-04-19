"""
- 数値型変換
- 対数化
- カテゴリ化
- 正規化
- 外れ値除去
- 次元圧縮
- 補完
"""
import pandas as pd
import numpy as np
# from fancyimpute import MICE   # fancyimpute requires Microsoft Visual C++ Build Tools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.datasets import load_boston
from seaborn import load_dataset

boston = load_boston()
df = pd.DataFrame(data=np.c_[boston['data'], boston['target']],
                  columns=np.append(boston['feature_names'], 'price'))

mpg = load_dataset('mpg')
"-------------------------------------------------------"
# 数値型変換

# 単一の値の型変換
# データ型の確認
type(40000 / 3)

# 整数型へ変換
int(40000 / 3)

# 浮動小数点型へ変換
float(40000 / 3)

# dfオブジェクトの型変換
# データ型の確認
df = pd.DataFrame({'value': [40000 / 3]})
df.dtypes

# データ型指定
df['value'].astype(int)  # int64
df['value'].astype(float)  # float64

# bit明示的な変換
# サポートされているbit数は以下の通り
# int/uint 8,16,32,64
# float      16,32,64
df['value'].astype('int8')
df['value'].astype('float64')

"-------------------------------------------------------"
# 対数化
mpg['horsepower_log'] = \
  mpg['horsepower'].apply(lambda x: np.log10(x / 1000 + 1))
"-------------------------------------------------------"
# カテゴリ変数化
mpg['horsepower_rank'] = \
  pd.Categorical(np.floor(mpg['horsepower']/50)*50).astype('category')

"-------------------------------------------------------"
# 正規化
# 少数点以下を扱えるようにするためfloat型に変換
mpg['weight'] = mpg['weight'].astype(float)

# 正規化を行うオブジェクトを生成
ss = StandardScaler()

# fit_transform関数は、fit関数（正規化するための前準備の計算）と
# transform関数（準備された情報から正規化の変換処理を行う）の両方を行う
result = ss.fit_transform(mpg[['weight', 'horsepower']])

mpg['weight_normalized'] = [x[0] for x in result]
mpg['horsepower_normalized'] = [x[1] for x in result]


"-------------------------------------------------------"
# 外れ値除去（平均値±3σ以内のデータのみ抽出）
reserve_tb = mpg[
  (abs(mpg['displacement'] - np.mean(mpg['displacement'])) /
   np.std(mpg['displacement']) <= 3)
].reset_index()

"-------------------------------------------------------"
# 次元圧縮

# n_componentsに、主成分分析で変換後の次元数を設定
pca = PCA(n_components=2)

# 主成分分析を実行
# pcaに主成分分析の変換パラメータが保存され、返り値に主成分分析後の値が返される
pca_values = pca.fit_transform(mpg[['weight', 'displacement']])

# 累積寄与率と寄与率の確認
print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))

# predict関数を利用し、同じ次元圧縮処理を実行
pca_newvalues = pca.transform(mpg[['weight', 'displacement']])

"-------------------------------------------------------"
"-------------------------------------------------------"
# 補完/欠損値処理
"-------------------------------------------------------"
df = pd.DataFrame(data=np.c_[boston['data'], boston['target']],
                  columns=np.append(boston['feature_names'], 'price'))
# 特定の値で補完
# replace関数によって、Noneをnanに変換
# （Noneを指定する際には文字列として指定する必要がある）
df.replace('None', np.nan, inplace=True)

# dropna関数によって、priceにnanを含むレコードを削除
df.dropna(subset=['price'], inplace=True)

# fillna関数によって、thicknessの欠損値を1で補完
df['price'].fillna(1, inplace=True)

"-------------------------------------------------------"
# 平均値で補完
# priceの平均値を計算
price_mean = df['price'].astype('float64').mean()

# priceの欠損値をpriceの平均値で補完
df['price'].fillna(price_mean, inplace=True)

"-------------------------------------------------------"
# 連鎖式による多重代入法
# replace関数によって、Noneをnanに変換
"""
production_miss_num = load_production_missing_num()
production_miss_num.replace('None', np.nan, inplace=True)

# mice関数を利用するためにデータ型を変換（mice関数内でモデル構築をするため）
production_miss_num['thickness'] = production_miss_num['thickness'].astype('float64')
production_miss_num['type'] = production_miss_num['type'].astype('category')
production_miss_num['fault_flg'] = production_miss_num['fault_flg'].astype('category')

# ダミー変数化
production_dummy_flg = pd.get_dummies(production_miss_num[['type', 'fault_flg']], drop_first=True)

# mice関数にPMMを指定して、多重代入法を実施
# n_imputationsは取得するデータセットの数
# n_burn_inは値を取得する前に試行する回数
mice = MICE(n_imputations=10, n_burn_in=50, impute_type='pmm')

# 処理内部でTensorFlowを利用
production_mice = mice.multiple_imputations(
  # 数値の列とダミー変数を連結
  pd.concat([production_miss_num[['length', 'thickness']],
             production_dummy_flg], axis=1)
)

# 下記に補完する値が格納されている
production_mice[0]
"""