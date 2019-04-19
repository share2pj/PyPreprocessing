"""
- 横持変換
- マトリックス変換
"""

import pandas as pd
from scipy.sparse import csc_matrix
from seaborn import load_dataset
mpg = load_dataset('mpg')
mpg['id'] = mpg.index
"-------------------------------------------------------"
# 横持変換
# 過去の予約一覧が含まれたreserve_tbを基に、顧客毎・人数別の予約回数を出力

# 元データイメージ
#       mpg  cylinders  displacement  horsepower  weight  acceleration  model_year  origin                     name  id
# 0    18.0          8         307.0       130.0    3504          12.0     70        usa  chevrolet chevelle malibu  0
# 1    15.0          8         350.0       165.0    3693          11.5     70        usa          buick skylark 320  1
# 2    18.0          8         318.0       150.0    3436          11.0     70        usa         plymouth satellite  1

# 出力イメージ
# cylinders  3   4  5   6    8
# origin
# europe     0  63  3   4    0
# japan      4  69  0   6    0
# usa        0  72  0  74  103


# pivot_table関数で、横持ち変換と集約処理を同時実行
# aggfuncに予約数をカウントする関数を指定
pd.set_option('max_columns', 10)
df = pd.pivot_table(mpg, index='origin', columns='cylinders',
                    values='id',
                    aggfunc=lambda x: len(x), fill_value=0)
"-------------------------------------------------------"
# マトリックス変換

# origin／cylinders別の車種表を生成
cnt_tb = mpg \
    .groupby(['origin', 'cylinders'])['id'].size() \
    .reset_index()
cnt_tb.columns = ['origin', 'cylinders', 'type_cnt']

# sparseMatrixの行／列に該当する列の値をカテゴリ型に変換
origin_id = pd.Categorical(cnt_tb['origin'])
cylinders_num = pd.Categorical(cnt_tb['cylinders'])

# スパースマトリックスを生成
# 1の引数は、指定した行列に対応した値、行番号、列番号の配列をまとめたタプルを指定
# shapeには、スパースマトリックスのサイズを指定（行数／列数のタプルを指定）
# （customer_id.codesはインデックス番号の取得）
# （len(customer_id.categories)は、customer_idのユニークな数を取得）
csc_matrix((cnt_tb['type_cnt'], (origin_id.codes, cylinders_num.codes)),
           shape=(len(origin_id.categories), len(cylinders_num.categories)))
"-------------------------------------------------------"
