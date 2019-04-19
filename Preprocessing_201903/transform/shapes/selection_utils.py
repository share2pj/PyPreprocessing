"""
- 列抽出
- 列削除
- 条件を満たす行取得
- サンプリング
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                  columns=diabetes['feature_names'] + ['target'])
df['id'] = df.index
"-------------------------------------------------------"
# 列抽出

# loc関数の2次元配列の2次元目に抽出したい列名の配列を指定することで、列を抽出
target_columns = \
    ['Age', 'Sex', 'bmi', 'bp', 's1', 's2', 's3']
df.loc[:, target_columns]

"-------------------------------------------------------"
# 列削除

# drop関数によって、不要な列を削除
# axisを1にすることによって、列の削除を指定
# inplaceをTrueに指定することによって、reserve_tbの書き換えを指定
delete_columns = ['s4', 's5']
df.drop(delete_columns, axis=1, inplace=False)

# inplaceをTrueに指定することによって、reserve_tbの書き換えを指定
df.drop(['s5', 's6'], axis=1, inplace=True)
"-------------------------------------------------------"
# 条件を満たす行を抽出

# query関数によって、SQLのようにDataFrameにアクセス
# bmi列が特定の範囲にあるレコードのみ抽出
df.query('0.02 <= bmi <= 0.025')
"-------------------------------------------------------"
# サンプリング

# 全データから比率を指定して抽出
# dfから50%サンプリング
df.sample(frac=0.5)


# 指定した列から指定した、一部のデータのみ抽出
# - サンプリングした顧客IDに関するデータを抽出

# df['id']に対して、unique()をかけて、重複を排除したidを取得
# sample関数によって、さらに顧客IDのサンプリングを実施
target = pd.Series(df['id'].unique()).sample(frac=0.5)

# isin関数によって、idがサンプリングした顧客IDのいずれかに一致した行を抽出
df[df['id'].isin(target)]
"-------------------------------------------------------"
