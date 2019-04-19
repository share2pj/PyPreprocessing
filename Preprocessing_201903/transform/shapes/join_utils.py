"""
- マスタテーブルとの結合
"""
import pandas as pd
import numpy as np
# ガベージコレクション(必要ないメモリの解放)のためのライブラリ
import gc

df_reviews = pd.read_csv('../../data/Datafiniti_Hotel_Reviews.csv')
df_hotels = pd.read_csv('../../data/hotels.csv')
"-------------------------------------------------------"
# マスタテーブルとの結合

# 特定の条件を満たすデータのみ抽出した上で、既存のkeyを用いて結合
# df_hotelsとdf_reviewsを、nameが等しいもの同士で内部結合
# countryがUSのデータのみ抽出
pd.merge(df_reviews.query('country == "US"'),
         df_hotels,
         on='name', how='inner')
"-------------------------------------------------------"
#
# area_nameごとにホテル数をカウント
small_area_mst = df_hotels.groupby(['province', 'city'], as_index=False).size().reset_index()
small_area_mst.columns = ['province', 'city', 'hotel_cnt']

# 20件以上であればjoin_area_idをsmall_area_name、以下ならばbig_area_nameとして設定（-1は、自ホテルを引いている）
small_area_mst['join_area_id'] = np.where(small_area_mst['hotel_cnt'] - 1 >= 20,
                                          small_area_mst['city'],
                                          small_area_mst['province'])

# 必要なくなった列を削除
small_area_mst.drop(['hotel_cnt', 'province'], axis=1, inplace=True)

# レコメンド元になるホテルにsmall_area_mstを結合することで、join_area_idを設定
base_hotel_mst = pd.merge(df_hotels, small_area_mst, on='city') \
                   .loc[:, ['hotel_id', 'join_area_id']]

# 下記は必要に応じて、メモリを解放(必須ではないですがメモリ量に余裕のないときに利用)
del small_area_mst
gc.collect()

"-------------------------------------------------------"