"""
- カテゴリ型変換
- ダミー変数化
- カテゴリ値集約
- カテゴリ値数値化
- カテゴリ値補完
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_boston
from seaborn import load_dataset

boston = load_boston()
df = pd.DataFrame(data=np.c_[boston['data'], boston['target']],
                  columns=np.append(boston['feature_names'], 'price'))

mpg = load_dataset('mpg')
"-------------------------------------------------------"
# カテゴリ型変換
# originがjapanのときにTRUEとするブール型を追加
# このコードは、as.type関数を利用しなくてもブール型に変換
mpg[['made_in_japan']] = (mpg[['origin']] == 'japan').astype('bool')

# originをカテゴリ型に変換
mpg['made_in_japan'] = \
  pd.Categorical(mpg['origin'], categories=['japan', 'other'])

# astype関数でも変換可能
# mpg['origin'] = mpg['origin'].astype('category')

# インデックスデータはcodesに格納されている
mpg['made_in_japan'].cat.codes

# マスタデータはcategoriesに格納されている
mpg['made_in_japan'].cat.categories

"-------------------------------------------------------"
# ダミー変数化

# ダミー変数化する前にカテゴリ型に変換
mpg['sex'] = pd.Categorical(mpg['origin'])

# get_dummies関数によってsexをダミー変数化
# drop_firstをFalseにすると、カテゴリ値の全種類の値のダミーフラグを生成
dummy_vars = pd.get_dummies(mpg['origin'], drop_first=False)

"-------------------------------------------------------"
# カテゴリ値集約

# pd.Categoricalによって、category型に変換
mpg['horsepower_rank'] = \
  pd.Categorical(np.floor(mpg['horsepower']/50)*50)

# マスタデータに'200以上'を追加
mpg['horsepower_rank'].cat.add_categories(['200以上'], inplace=True)

# 集約するデータを書き換え
# category型は、=または!=の判定のみ可能なので、isin関数を利用
mpg.loc[mpg['horsepower_rank'].isin([200.0, 210.0, 220.0, 230.0]), 'horsepower_rank'] = '200以上'

# 利用されていないマスタデータを削除
mpg['horsepower_rank'].cat.remove_unused_categories(inplace=True)

"-------------------------------------------------------"
mpg['sex_and_age'] = pd.Categorical(
  # 連結する列を抽出
  mpg[['origin', 'cylinders']]

    # lambda関数内でoriginと2区切りのcylindersを_を挟んで文字列として連結
    .apply(lambda x: '{}_{}'.format(x[0], np.floor(x[1] / 2) * 2),
           axis=1)
)

"-------------------------------------------------------"
# カテゴリ値数値化
"""
# 製品種別ごとの障害数
fault_cnt_per_type = production \
  .query('fault_flg') \
  .groupby('type')['fault_flg'] \
  .count()

# 製品種別ごとの製造数
type_cnt = production.groupby('type')['fault_flg'].count()

production['type_fault_rate'] = production[['type', 'fault_flg']] \
  .apply(lambda x:
         (fault_cnt_per_type[x[0]] - int(x[1])) / (type_cnt[x[0]] - 1),
         axis=1)
"""
"-------------------------------------------------------"
# カテゴリ値補完
"""
# replace関数によって、Noneをnanに変換
mpg.replace('None', np.nan, inplace=True)

# 欠損していないデータの抽出
train = mpg.dropna(subset=['origin'], inplace=False)

# 欠損しているデータの抽出
test = mpg \
  .loc[mpg.index.difference(train.index), :]

# knnモデル生成、n_neighborsはknnのkパラメータ
kn = KNeighborsClassifier(n_neighbors=3)

# knnモデル学習
kn.fit(train[['mpg', 'horsepower']], train['origin'])

# knnモデルによって予測値を計算し、mpgを補完
test['origin'] = kn.predict(test[['mpg', 'horsepower']])
"""
"-------------------------------------------------------"
