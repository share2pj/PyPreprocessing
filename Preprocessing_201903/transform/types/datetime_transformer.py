"""
- 日時型処理
- 日時、日付型変換
- 年月日、時刻、曜日変換
- 日時差計算
- 季節変換
- Timezone変換
- 平日休日判定
"""
import pandas as pd
import datetime

df_reviews = pd.read_csv('../../data/Datafiniti_Hotel_Reviews.csv')
"-------------------------------------------------------"
# 日時型処理
# to_datetime関数で、datetime64[ns]型に変換
pd.to_datetime(df_reviews['dateUpdated'], format='%Y-%m-%d %H:%M:%S')

# datetime64[ns]型から日付情報を取得
pd.to_datetime(df_reviews['dateUpdated'],
               format='%Y-%m-%d %H:%M:%S').dt.date
pd.to_datetime(df_reviews['dateUpdated'], format='%Y-%m-%d').dt.date
"-------------------------------------------------------"
# 日時、日付型変換
# dateUpdatedをdatetime64[ns]型に変換
# unitの標準はns(nano seconds) argsに指定することでms, usなど使用可能だが、する場面はほぼないのでは
df_reviews['dateUpdated'] = pd.to_datetime(df_reviews['dateUpdated'], format='%Y-%m-%d %H:%M:%S')


# datetimeに変換すると、Series.dtの各パラメータから値を取得可能
# year, month, day, dayofweek, hour, minute, second
df_reviews['dateUpdated'].dt.year  # 年を取得
df_reviews['dateUpdated'].dt.dayofweek  # 曜日（0=日曜日、1＝月曜日）を数値で取得
df_reviews['dateUpdated'].dt.second  # 時刻の秒を取得

# 指定したフォーマットの文字列に変換
df_reviews['dateUpdated'].dt.strftime('%Y-%m-%d %H:%M:%S')
"-------------------------------------------------------"
# 日時差計算
# dateUpdated, dateUpdatedをdatetime64[ns]型に変換
df_reviews['dateUpdated'] = pd.to_datetime(df_reviews['dateUpdated'], format='%Y-%m-%d %H:%M:%S')
df_reviews['dateAdded'] = pd.to_datetime(df_reviews['dateAdded'], format='%Y-%m-%d %H:%M:%S')

# 年の差分を計算（月以下の日時要素は考慮しない）
df_reviews['dateUpdated'].dt.year - df_reviews['dateAdded'].dt.year

# 月の差分を取得（日以下の日時要素は考慮しない）
(df_reviews['dateUpdated'].dt.year * 12 + df_reviews['dateUpdated'].dt.month) \
 - (df_reviews['dateAdded'].dt.year * 12 + df_reviews['dateAdded'].dt.month)

# 日単位で差分を計算
(df_reviews['dateUpdated'] - df_reviews['dateAdded']).astype('timedelta64[D]')

# 時単位で差分を計算
(df_reviews['dateUpdated'] - df_reviews['dateAdded']).astype('timedelta64[h]')

# 分単位で差分を計算
(df_reviews['dateUpdated'] - df_reviews['dateAdded']).astype('timedelta64[m]')

# 秒単位で差分を計算
(df_reviews['dateUpdated'] - df_reviews['dateAdded']).astype('timedelta64[s]')

"-------------------------------------------------------"
# 日時計算
# dateUpdatedをdatetime64[ns]型に変換
df_reviews['dateUpdated'] = pd.to_datetime(df_reviews['dateUpdated'], format='%Y-%m-%d %H:%M:%S')

# timedeltaはweeks, days, hours, minutes, seconds, microseconds, millisecondsをサポート（年月は非対応）
df_reviews['dateUpdated'] + datetime.timedelta(days=1)
df_reviews['dateUpdated'] + datetime.timedelta(hours=1)
df_reviews['dateUpdated'] + datetime.timedelta(minutes=1)
df_reviews['dateUpdated'] + datetime.timedelta(seconds=1)
"-------------------------------------------------------"
# 季節変換（カテゴリ文字列に変換する）
# dateUpdatedをdatetime64[ns]型に変換
df_reviews['dateUpdated'] = pd.to_datetime(df_reviews['dateUpdated'], format='%Y-%m-%d %H:%M:%S')


# 月の数字を季節に変換する関数
def to_season(month_num):
    season = 'winter'
    if 3 <= month_num <= 5:
        season = 'spring'
    elif 6 <= month_num <= 8:
        season = 'summer'
    elif 9 <= month_num <= 11:
        season = 'autumn'
    return season


# 季節に変換
df_reviews['update_season'] = pd.Categorical(
    df_reviews['dateUpdated'].dt.month.apply(to_season),
    categories=['spring', 'summer', 'autumn', 'winter']
)
"-------------------------------------------------------"
"""
# 平日休日判定
# 休日マスタと結合 holiday_flagにtrue/falseが入っているので、その結果に応じてフィルタリング
pd.merge(df_reviews, holiday_mst,
         left_on='dateUpdated', right_on='target_day')
"""
"-------------------------------------------------------"

