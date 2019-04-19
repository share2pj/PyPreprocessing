"""
- 検証用のデータ分割
- ホールドアウト検証
- k分割交差検証
- rolling window検証
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                  columns=diabetes['feature_names'] + ['target'])
df['id'] = df.index
"-------------------------------------------------------"
# ホールドアウト検証用のデータ分割

x = df.drop('target', axis=1)
y = df[['target']]

# 予測モデルの入力値と予測対象の値を別々にtrain_test_split関数に設定、test_sizeは検証データの割合
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 作成したdfすべてについて、行名を現在の行番号に直す
df_list = ['x_train', 'x_test', 'y_train', 'y_test']

for name in df_list:
    vars()[name].reset_index(inplace=True, drop=True)

# trainを学習データ、testを検証データとして機械学習モデルの構築、検証をするがここでは割愛

"-------------------------------------------------------"
# 交差検証 - k分割交差検証用のデータ分割

k_fold = KFold(n_splits=4, shuffle=True)

# 対象の行番号リストを生成
row_no_list = list(range(len(y_train)))


# 交差数分繰り返し処理、並列処理も可能な部分
# x_train, y_trainから、k分割したtrain/testを新たに生成する
# cv = cross validation
# k_fold関数を使うと、n_splitsで定義した個数のtrain - testの組み合わせリストが生成される(下のコードは、4周する)
for train_cv_no, test_cv_no in k_fold.split(row_no_list):
    # 交差検証におけるデータを抽出
    x_train_cv, x_test_cv = x_train.iloc[train_cv_no, :], x_train.iloc[test_cv_no, :]
    y_train_cv, y_test_cv = y_train.iloc[train_cv_no, :], y_train.iloc[test_cv_no, :]

    # 本来は機械学習モデルの構築、検証をするがここでは割愛
# 本来は交差検証の結果をまとめをするがここでは割愛
"-------------------------------------------------------"
# 交差検証 - rolling windowのデータ分割(time-series分析の際などに使われる)

# 元データをid順に並び替え
df.sort_values(by='id')

# rolling windowのパラメータ設定
train_window_start = 1  # 開始行番号
train_window_end = 24  # 終了行番号を指定
horizon = 12  # 検証データのデータ数を指定
skip = 12  # skipにスライドするデータ数を設定

data_end = len(df.index)  # データの末尾の値を、処理の終了判定のために取得

while True:
    # 検証データの終了行番号を計算
    test_window_end = train_window_end + horizon

    # 行番号を指定して、元データから学習データを取得
    # train_window_startの部分を1に固定すれば、学習データを増やしていく検証に変更可能
    train = df[train_window_start:train_window_end]

    # 行番号を指定して、元データから検証データを取得
    test = df[(train_window_end + 1):test_window_end]

    # 検証データの終了行番号が元データの行数以上になっているか判定、全データを対象にした場合終了
    if test_window_end >= data_end:
        break

    # 本来は機械学習モデルの構築、検証をするがここでは割愛
    # データをスライドさせる
    train_window_start += skip
    train_window_end += skip

# 交差検定の結果をまとめる

"-------------------------------------------------------"
