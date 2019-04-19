"""
- imbalanced-learning: オーバーサンプリング/アンダーサンプリング　
"""
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
"-------------------------------------------------------"
# オーバーサンプリング

# SMOTE関数の設定
# ratioは不均衡データにおける少ない例のデータを多い方のデータの何割まで増やすか設定
# （autoの場合は同じ数まで増やす、0.5と設定すると5割までデータを増やす）
# k_neighborsはsmoteのkパラメータ
# random_stateは乱数のseed 結果を固定したい場合には適当なseedを設定する
sm = SMOTE(ratio='auto', k_neighbors=5, random_state=71)
balance_data, balance_target = sm.fit_sample(df[['sepal length', 'sepal width']],
                                             df['target'])

print(len(balance_data), len(balance_target))
"-------------------------------------------------------"
