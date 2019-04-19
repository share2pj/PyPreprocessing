"""
- 形態素解析による分解
- ストップワード除去
- 集合データ、ベクトル変換
- 重要度調整

- 文字列スプリットとか置換とかなにもやらないのな
"""
import os
# Mecabについて
# ライブラリが存在しない場合、natto ではなくnatto-pyをinstallしてください
#
from natto.mecab import MeCab
from gensim import corpora, matutils, models  # bug of wordsを作成するためのライブラリ読み込み

txt_dir = os.path.abspath(os.path.dirname(__file__) + '/../../data/txt')
files = os.listdir(txt_dir)
print(files)
with open(txt_dir + '/meros.txt', 'r', encoding="utf-8") as f:
    txt = f.read()

"-------------------------------------------------------"
# 形態素解析による分解
# merosには、メロスの文章データが格納
# MeCabを実行するオブジェクトを生成
mc = MeCab()

# MeCabを用いて、形態素解析を実行
# テキストに含まれる単語リストを返却する関数
def word_list_create(txt):
    tmp_list = []
    for part_and_word in mc.parse(txt, as_nodes=True):
        # 形態素解析結果のpart_and_wordが開始/終了オブジェクトでないことを判定
        if not (part_and_word.is_bos() or part_and_word.is_eos()):
            # 形態素解析結果から品詞と単語を取得
            part, word = part_and_word.feature.split(',', 1)

            # 名詞と動詞の単語を抽出
            if part == '名詞' or part == '動詞':
                tmp_list.append(part_and_word.surface)
    return tmp_list


word_list_create(txt)
"-------------------------------------------------------"
# ストップワード除去
# 集合データ、ベクトル変換
txt_word_list = []

# フォルダ配下のテキストファイルを1つずつ読み込み
for file in files:
    print(file)
    with open(os.path.dirname(__file__) + '/txt/'+file, 'r') as f:
        txt = f.read()

    # 単語リストを作成し、テキストファイルごとの単語リストに追加
    txt_word_list.append(word_list_create(txt))

# bug of wordsを作成するため全種類の単語を把握し、単語IDを付与した辞書を作成
corpus_dic = corpora.Dictionary(txt_word_list)

# 各文章の単語リストをコーパス（辞書の単語IDと単語の出現回数）リストに変換
corpus_list = [corpus_dic.doc2bow(word_in_text) for word_in_text in txt_word_list]

# コーパスリストをスパースマトリックス（csc型）に変換
word_matrix = matutils.corpus2csc(corpus_list)

"-------------------------------------------------------"
# 重要度調整
# 上で作成したcorpus_listを基にTF-IDFのモデルを生成
tfidf_model = models.TfidfModel(corpus_list, normalize=True)

# corpusにTF-IDFを適用
corpus_list_tfidf = tfidf_model[corpus_list]
word_matrix = matutils.corpus2csc(corpus_list_tfidf)
"-------------------------------------------------------"
