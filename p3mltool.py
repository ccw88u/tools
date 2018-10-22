import jieba
import jieba.posseg
import numpy as np

from sklearn.externals import joblib

def jeiba_cut(cons, POSswitch=0):
    if POSswitch == 0:
        rs = ' '.join(jieba.cut(dropuselesschars(cons)))
    else:
        lst = jieba.posseg.lcut(dropuselesschars(cons))
        # pair('哀怨', 'v'), pair('說', 'v')
        aplst = []
        POSlst = []
        for n, pos in lst:
            if pos in POSlst:
                aplst.append(n)
        rs = ' '.join(aplst)
    return rs

def dropuselesschars(rs):
    import re

    replace_re = [
        '^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?']
    for rev in replace_re:
        rs = re.sub(r'%s' % rev, '', rs, re.I)

    # . 不要去掉，因為有網址問題
    rs = re.sub("[\s+\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%&*（）／]+", "", rs)

    # . 不要去掉，因為有網址問題
    # 空白不能移除
    dropchar = ['？', '…', '，', '。', '！', '、', '《', '》', '；', ';', '「', '」', '「', '」', '（', '）', '(', ')', '?',
                '〈', '〉', '&nbsp;', '&nbsp', 'nbsp', '蘋果報導', '報導', '綜合', '╱', '】', '【', '：', ':', '╱']

    # remove 00~99  如年紀、金額、日期、重量、時間
    for x in range(0, 100):
        dropchar.append('%02d' % x)

    for dc in dropchar:
        rs = rs.replace(dc, '')

    return rs

def jeiba_dostop(text, stopwordfile='stopword_chi.txt', translate=''):
    # 存停用詞, 分詞, 過濾後分詞的list
    stopWords = []
    segments = []
    remainderWords = []

    # 讀入停用詞檔
    with open(stopwordfile, 'r', encoding='UTF-8') as file:
        for data in file.readlines():
            data = data.strip()
            stopWords.append(data)
    text = dropuselesschars(text)
    segments = jieba.cut(text)
    # segments = jieba.cut_for_search(text)         ## no good
    remainderWords = list(
        filter(lambda a: a not in stopWords and a != '\n', segments))
    remainderWords = ' '.join(remainderWords)
    return remainderWords

def ml_module_load(fp):
    return joblib.load(fp)

def ml_module_save(X, fp):
    joblib.dump(X, fp)

# 載入model yaml & model weight
def load_model_yaml_weight(yaml_path, weight_path):
    from keras.models import model_from_yaml
    with open(yaml_path) as yamlfile:
        loaded_model_yaml = yamlfile.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights(weight_path)
    return model

def random_split(all_df, spn=0.8):
    msk = np.random.rand(len(all_df)) < 0.8
    train_df = all_df[msk]
    test_df = all_df[~msk]
    return train_df, test_df

# 打亂資料
def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]
    return x, t

def getcs_score(cs, articleid):
    for pos in cs[articleid].argsort()[::-1][1:]:
        score = cs[articleid][pos]
    return score

# 讀取圖檔尺寸 & 看圖檔
def read_img_manview(filepath, shapearg=1, imgshowarg=1):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import os
    if os.path.isfile(filepath):
        img1 = mpimg.imread(filepath)
        if shapearg == 1: print(img1.shape)
        if imgshowarg == 1:
            plt.imshow(img1)
            plt.show()
    else:
        print('%s is not exist')
