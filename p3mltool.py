import jieba
import jieba.posseg
import numpy as np

from sklearn.externals import joblib
from PIL import Image

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

# softmax predict 最高分數類別及信心值
def model_predict_class_confidence(result):
    pred_classes = result.argmax(axis=-1)
    pred_class_label = pred_classes[0]
    pred_confidence = result[0][pred_class_label]
    return pred_class_label, pred_confidence

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


# 讀取圖檔，並直接resize--PIL Image
def resize_image_imgarray(imgpath, rswidth=64, rsheight=64):
    from PIL import Image
    img = Image.open(imgpath)
    img = img.resize((rswidth, rsheight), resample=Image.BILINEAR)
    return img

# 讀取圖檔，並直接resize--opencv
def resize_image_imgarray_cv2(imgpath, rswidth=64, rsheight=64):
    import cv2
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    ##寬，高
    img = cv2.resize(img, (rsheight, rswidth), interpolation=cv2.INTER_CUBIC)    
    return img

# 變更圖檔大小後，直接進行檔案儲存
def resize_image(imgpath, savepath, rswidth=64, rsheight=64):
    img = Image.open(imgpath)
    img = img.resize((rswidth, rsheight), resample=Image.BILINEAR)
    img.save(savepath)

# 批次讀取某個路徑下圖檔及其onehot label
# load_data('./dataset/test/')
def load_dataload_data(dirpath, imgfmt='jpg', img_w=256, img_h=256, readmax=1000000):
    import os
    if dirpath[-1] != os.sep: dirpath += os.sep
    files = os.listdir(dirpath)
    files.sort()
    images = []
    labels = []
    labeint = 0
    for filedir in files:
        picfiles = glob.glob('%s%s/*.%s' % (dirpath, filedir, imgfmt))
        for f in picfiles:
            # print('f', f)
            img_path = f
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            images.append(img_array)

            labels.append(labeint)
        labeint += 1
    data = np.array(images)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels, labeint)
    return data, labels


# include imagenet top, transfer learning build model
def build_model_transfer(application, img_w, img_h, num_class, lastdensesize=512):
    ##shape can not smaller then 
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications import densenet       ## densenet121, densenet169, densenet201
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.xception import Xception
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg19 import VGG19
    from keras.applications.resnet50 import ResNet50
    from keras.applications.mobilenet import MobileNet
    from keras.applications.nasnet import NASNet  ## NASNetLarge, NASNetMobile
    
    input_tensor = Input(shape=(img_w, img_h, 3))

    if application == 'inceptionv3':
        base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
    # smallest: 48 * 48 default: 224 * 224    
    elif application == 'vgg16':
        base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=True)
    elif application == 'vgg19':
        base_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=True)
    # smallest: 197 * 197 default: 224 * 224     
    elif application == 'resnet50':
        base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=True)
    # smallest: 32 * 32 default: 224 * 224    
    elif application == 'mobilenet':
        base_model = MobileNet(input_tensor=input_tensor, weights='imagenet', include_top=True)
    elif application == 'Xception':
        base_model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=True)
    elif application == 'densenet121':
        base_model = densenet.DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=True)
    elif application == 'densenet169':
        base_model = densenet.DenseNet169(input_tensor=input_tensor, weights='imagenet', include_top=True)
    elif application == 'densenet201':
        base_model = densenet.DenseNet201(input_tensor=input_tensor, weights='imagenet', include_top=True)
    elif application == 'nasnetlarge':
        base_model = densenet.NASNetLarge(input_tensor=input_tensor, weights='imagenet', include_top=True)
    elif application == 'nasnetmobile':
        base_model = densenet.NASNetMobile(input_tensor=input_tensor, weights='imagenet', include_top=True)
    elif application == 'inceptionresnetv2':
         base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=True)
    
    x = base_model.output
    x = Dense(lastdensesize, activation='relu')(x)
    # 最後一層不要 batch normalize / dropout
    outputs = Dense(num_class, activation='softmax')(x)    
    model = Model(base_model.inputs, outputs)
    #model.summary()
    
    return model

# 由漢語拼音(有音調) 取得該字注音音調
def checktone(wordpinyin):
    # 部分符合該字為1聲
    tone1words = ['ā', 'ē', 'ī', 'ō', 'ū']
    # 全部符合該字為1聲
    tone1fulls = ['a', 'ba', 'de', 'ge', 'guo', 'hei', 'la', 'le', 'li', 'lou', 'ma', 'men', 'ne', 'wa', 'ya',
                  'zhe', 'shi', 'me', 'da', 'shang', 'luo', 'zi', 'fu', 'di', 'he', 'huo', 'mo', 'bo', 'hu']
    # 部分符合該字為2聲
    tone2words = ['á', 'é', 'í', 'ó', 'ú', 'ǘ', 'ḿ', 'ń']  
    # 部分符合該字為2聲
    tone3words = ['ǎ', 'ě', 'ǐ', 'ǒ', 'ǔ', 'ǚ']
    tone4words = ['à', 'è', 'ì', 'ò', 'ù', 'ǹ', 'ǜ']
    
    # print('wordpinyin', wordpinyin)
    findtone = -1    
    tone1find = False
    for tv in tone1words:
        if wordpinyin.find(tv) != -1:
            findtone = 1
            tone1find = True
    for tv in tone1fulls:
        if wordpinyin == tv:
            findtone = 1
            tone1find = True

    tone2find = False
    for tv in tone2words:
        if wordpinyin.find(tv) != -1:
            findtone = 2
            tone2find = True

    tone3find = False
    for tv in tone3words:
        if wordpinyin.find(tv) != -1:
            findtone = 3
            tone3find = True

    tone4find = False
    for tv in tone4words:
        if wordpinyin.find(tv) != -1:
            findtone = 4
            tone4find = True
    
    # 找到兩個音, 代表有問題
    Twotonefinded = False
    if tone1find == tone2find == True or tone2find == tone3find == True or tone3find == tone4find == True \
        or tone1find == tone4find == True:
        Twotonefinded = True

    return Twotonefinded, findtone

