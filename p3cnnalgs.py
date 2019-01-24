from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers import Dense, Dropout, Lambda, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model

from keras.layers import Input, add


# -------------------simp CNN{S}-------------------
def build_model():
    # 建立簡單的線性執行的模型
    model = Sequential()
    # 建立卷積層，filter=32,即 output size, Kernal Size: 2x2, activation function 採用 relu
    ## old : input_shape=(20, 11, 3)
    ## new : input_shape=(140, 257, 3)
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(140, 257, 3)))
    # 建立池化層，池化大小=2x2，取最大值
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
    model.add(Dropout(0.5))
    # Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
    model.add(Flatten())
    # 全連接層: 128個output
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # Add output layer
    model.add(Dense(tone_num, activation='softmax'))
    return model

def build_model(img_h, img_w, tone_num):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(img_h, img_w, 1),
                activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(tone_num, activation='softmax'))
    return model
# -------------------simp CNN{E}-------------------


# -------------------VGG16{S}-------------------
def build_VGG16(img_h, img_w, nb_classes, channels=3):
    INPUTSIZE = (img_h, img_w, channels)
    
    model = Sequential()  
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=INPUTSIZE,padding='same',activation='relu',kernel_initializer='random_normal'))  
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Flatten())  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(nb_classes,activation='softmax'))
    model.summary()
    
    return model
## way to use VGG16
## min-size: smallest: 48 * 48 default: 224 * 224 
## model = build_VGG16(img_w, img_h, tone_num, channels=1)      
# -------------------VGG16{E}-------------------


# -------------------VGG19{S}-------------------
def build_VGG19(img_h, img_w, nb_classes, channels=3):
    
    INPUTSIZE = (img_h, img_w, channels)
    model = Sequential()
    
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=INPUTSIZE,padding='same',activation='relu',kernel_initializer='random_normal'))  
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))  
    
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))  
    
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))  
    
    model.add(Flatten())  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(nb_classes,activation='softmax'))
    model.summary()
    
    return model

## way to use VGG19
## min-size: smallest: 48 * 48 default: 224 * 224 
## model = build_VGG19(img_w, img_h, tone_num, channels=1)    
# -------------------VGG19{E}-------------------


# -------------------resnet50{S}-------------------


def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  
  
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
    x = Conv2d_BN(inpt,nb_filter=nb_filter[0],kernel_size=(1,1),strides=strides,padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3), padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')  
    if with_conv_shortcut:  
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)  
        x = add([x,shortcut])  
        return x  
    else:  
        x = add([x,inpt])  
        return x  

def resnet50(img_h, img_w, nb_classes, channels=3):

	INPUTSIZE = (img_h, img_w, channels)

    img_input = Input(shape=INPUTSIZE)  
    x = ZeroPadding2D((3,3))(img_input)  
    x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')  
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
  
    x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3),strides=(1,1),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))  
  
    x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))  
  
    x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))  
  
    x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3))  
    x = AveragePooling2D(pool_size=(7,7),padding='same')(x)  
    x = Flatten()(x)
    output = Dense(tone_num,activation='softmax')(x)

    model=Model(input=img_input, output=output)
    model.summary()

    return model

# -------------------resnet50{E}-------------------

from keras import layers
from keras.layers import GlobalMaxPooling2D

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def inceptionv3(img_h, img_w, nb_classes, channels=3):
    channel_axis = 3   # channels last

    INPUTSIZE = (img_h, img_w, channels)
    img_input = Input(shape=INPUTSIZE)  

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    inputs = img_input
        
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    return model

# -------------------inception_resnet_v1{S}-------------------

RESNET_V1_A_COUNT = 0
RESNET_V1_B_COUNT = 0
RESNET_V1_C_COUNT = 0
 
 
def resnet_v1_stem(x_input):
    with K.name_scope('Stem'):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid')(x_input)
        x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)
        x = Conv2D(80, (1, 1), activation='relu', padding='same')(x)
        x = Conv2D(192, (3, 3), activation='relu', padding='valid')(x)
        x = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
    return x
 
 
def inception_resnet_v1_A(x_input, scale_residual=True):
    """ 35x35 卷积核"""
    global RESNET_V1_A_COUNT
    RESNET_V1_A_COUNT += 1
    with K.name_scope('resnet_v1_A' + str(RESNET_V1_A_COUNT)):
        ar1 = Conv2D(32, (1, 1), activation='relu', padding='same')(x_input)
 
        ar2 = Conv2D(32, (1, 1), activation='relu', padding='same')(x_input)
        ar2 = Conv2D(32, (3, 3), activation='relu', padding='same')(ar2)
 
        ar3 = Conv2D(32, (1, 1), activation='relu', padding='same')(x_input)
        ar3 = Conv2D(32, (3, 3), activation='relu', padding='same')(ar3)
        ar3 = Conv2D(32, (3, 3), activation='relu', padding='same')(ar3)
 
        merged_vector = concatenate([ar1, ar2, ar3], axis=-1)
 
        ar = Conv2D(256, (1, 1), activation='linear', padding='same')(merged_vector)
 
        if scale_residual:  # 是否缩小
            ar = Lambda(lambda x: 0.1*x)(ar)
        x = add([x_input, ar])
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
    return x
 
 
def inception_resnet_v1_B(x_input, scale_residual=True):
    """ 17x17 卷积核"""
    global RESNET_V1_B_COUNT
    RESNET_V1_B_COUNT += 1
    with K.name_scope('resnet_v1_B' + str(RESNET_V1_B_COUNT)):
        br1 = Conv2D(128, (1, 1), activation='relu', padding='same')(x_input)
 
        br2 = Conv2D(128, (1, 1), activation='relu', padding='same')(x_input)
        br2 = Conv2D(128, (1, 7), activation='relu', padding='same')(br2)
        br2 = Conv2D(128, (7, 1), activation='relu', padding='same')(br2)
 
        merged_vector = concatenate([br1, br2], axis=-1)
 
        br = Conv2D(896, (1, 1), activation='linear', padding='same')(merged_vector)
 
        if scale_residual:
            br = Lambda(lambda x: 0.1*x)(br)
        x = add([x_input, br])
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
 
    return x
 
 
def inception_resnet_v1_C(x_input, scale_residual=True):
    global RESNET_V1_C_COUNT
    RESNET_V1_C_COUNT += 1
    with K.name_scope('resnet_v1_C' + str(RESNET_V1_C_COUNT)):
        cr1 = Conv2D(192, (1, 1), activation='relu', padding='same')(x_input)
 
        cr2 = Conv2D(192, (1, 1), activation='relu', padding='same')(x_input)
        cr2 = Conv2D(192, (1, 3), activation='relu', padding='same')(cr2)
        cr2 = Conv2D(192, (3, 1), activation='relu', padding='same')(cr2)
 
        merged_vector = concatenate([cr1, cr2], axis=-1)
 
        cr = Conv2D(1792, (1, 1), activation='relu', padding='same')(merged_vector)
 
        if scale_residual:
            cr = Lambda(lambda x: 0.1*x)
        x = add([x_input, cr])
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
    return x
 
 
def reduction_resnet_A(x_input, k=192, l=224, m=256, n=384):
    with K.name_scope('reduction_resnet_A'):
        ra1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)
 
        ra2 = Conv2D(n, (3, 3), activation='relu', strides=(2, 2), padding='valid')(x_input)
 
        ra3 = Conv2D(k, (1, 1), activation='relu', padding='same')(x_input)
        ra3 = Conv2D(l, (3, 3), activation='relu', padding='same')(ra3)
        ra3 = Conv2D(m, (3, 3), activation='relu', strides=(2, 2), padding='valid')(ra3)
 
        merged_vector = concatenate([ra1, ra2, ra3], axis=-1)
 
        x = BatchNormalization(axis=-1)(merged_vector)
        x = Activation('relu')(x)
    return x
 
 
def reduction_resnet_B(x_input):
    with K.name_scope('reduction_resnet_B'):
        rb1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),padding='valid')(x_input)
 
        rb2 = Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
        rb2 = Conv2D(384, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb2)
 
        rb3 = Conv2D(256, (1, 1),activation='relu', padding='same')(x_input)
        rb3 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb3)
 
        rb4 = Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
        rb4 = Conv2D(256, (3, 3), activation='relu', padding='same')(rb4)
        rb4 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb4)
 
        merged_vector = concatenate([rb1, rb2, rb3, rb4], axis=-1)
 
        x = BatchNormalization(axis=-1)(merged_vector)
        x = Activation('relu')(x)
    return x
 
 
def inception_resnet_v1_backbone(img_h, img_w, nb_classes, channels=3,, scale=True):
    x_input = Input(shape=(img_h, img_w, channels))
    # stem
    x = resnet_v1_stem(x_input)
 
    # 5 x inception_resnet_v1_A
    for i in range(5):
        x = inception_resnet_v1_A(x, scale_residual=False)
 
    # reduction_resnet_A
    x = reduction_resnet_A(x, k=192, l=192, m=256, n=384)
 
    # 10 x inception_resnet_v1_B
    for i in range(10):
        x = inception_resnet_v1_B(x, scale_residual=True)
 
    # Reduction B
    x = reduction_resnet_B(x)
 
    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_v1_C(x, scale_residual=True)
 
    # Average Pooling
    x = AveragePooling2D(pool_size=(8, 8))(x)
 
    # dropout
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=nb_classes, activation='softmax')(x)
 
    return Model(inputs=x_input, outputs=x, name='Inception-Resnet-v1')
# -------------------inception_resnet_v1{E}-------------------


# -------------------inception_resnet_v2{S}-------------------

RESNET_V2_A_COUNT = 0
RESNET_V2_B_COUNT = 0
RESNET_V2_C_COUNT = 0
 
 
def resnet_v2_stem(x_input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''
 
    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    with K.name_scope("stem"):
        x = Conv2D(32, (3, 3), activation="relu", strides=(2, 2))(x_input)  # 149 * 149 * 32
        x = Conv2D(32, (3, 3), activation="relu")(x)  # 147 * 147 * 32
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)  # 147 * 147 * 64
 
        x1 = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x2 = Conv2D(96, (3, 3), activation="relu", strides=(2, 2))(x)
 
        x = concatenate([x1, x2], axis=-1)  # 73 * 73 * 160
 
        x1 = Conv2D(64, (1, 1), activation="relu", padding="same")(x)
        x1 = Conv2D(96, (3, 3), activation="relu")(x1)
 
        x2 = Conv2D(64, (1, 1), activation="relu", padding="same")(x)
        x2 = Conv2D(64, (7, 1), activation="relu", padding="same")(x2)
        x2 = Conv2D(64, (1, 7), activation="relu", padding="same")(x2)
        x2 = Conv2D(96, (3, 3), activation="relu", padding="valid")(x2)
 
        x = concatenate([x1, x2], axis=-1)  # 71 * 71 * 192
 
        x1 = Conv2D(192, (3, 3), activation="relu", strides=(2, 2))(x)
 
        x2 = MaxPooling2D((3, 3), strides=(2, 2))(x)
 
        x = concatenate([x1, x2], axis=-1)  # 35 * 35 * 384
 
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x
 
 
def inception_resnet_v2_A(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_A block which is a 35 * 35 grid module.'''
    global RESNET_V2_A_COUNT
    RESNET_V2_A_COUNT += 1
    with K.name_scope('inception_resnet_v2_A' + str(RESNET_V2_A_COUNT)):
        ar1 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
 
        ar2 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
        ar2 = Conv2D(32, (3, 3), activation="relu", padding="same")(ar2)
 
        ar3 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
        ar3 = Conv2D(48, (3, 3), activation="relu", padding="same")(ar3)
        ar3 = Conv2D(64, (3, 3), activation="relu", padding="same")(ar3)
 
        merged = concatenate([ar1, ar2, ar3], axis=-1)
 
        ar = Conv2D(384, (1, 1), activation="linear", padding="same")(merged)
        if scale_residual: ar = Lambda(lambda a: a * 0.1)(ar)
 
        x = add([x_input, ar])
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x
 
 
def inception_resnet_v2_B(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_B block which is a 17 * 17 grid module.'''
    global RESNET_V2_B_COUNT
    RESNET_V2_B_COUNT += 1
    with K.name_scope('inception_resnet_v2_B' + str(RESNET_V2_B_COUNT)):
        br1 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)
 
        br2 = Conv2D(128, (1, 1), activation="relu", padding="same")(x_input)
        br2 = Conv2D(160, (1, 7), activation="relu", padding="same")(br2)
        br2 = Conv2D(192, (7, 1), activation="relu", padding="same")(br2)
 
        merged = concatenate([br1, br2], axis=-1)
 
        br = Conv2D(1152, (1, 1), activation="linear", padding="same")(merged)
        if scale_residual: br = Lambda(lambda b: b * 0.1)(br)
 
        x = add([x_input, br])
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x
 
 
def inception_resnet_v2_C(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_C block which is a 8 * 8 grid module.'''
    global RESNET_V2_C_COUNT
    RESNET_V2_C_COUNT += 1
    with K.name_scope('inception_resnet_v2_C' + str(RESNET_V2_C_COUNT)):
        cr1 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)
 
        cr2 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)
        cr2 = Conv2D(224, (1, 3), activation="relu", padding="same")(cr2)
        cr2 = Conv2D(256, (3, 1), activation="relu", padding="same")(cr2)
 
        merged = concatenate([cr1, cr2], axis=-1)
 
        cr = Conv2D(2144, (1, 1), activation="linear", padding="same")(merged)
        if scale_residual: cr = Lambda(lambda c: c * 0.1)(cr)
 
        x = add([x_input, cr])
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x
 
 
def reduction_resnet_v2_B(x_input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_ResNet_B block.'''
    with K.name_scope('reduction_resnet_v2_B'):
        rbr1 = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x_input)
 
        rbr2 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
        rbr2 = Conv2D(384, (3, 3), activation="relu", strides=(2, 2))(rbr2)
 
        rbr3 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
        rbr3 = Conv2D(288, (3, 3), activation="relu", strides=(2, 2))(rbr3)
 
        rbr4 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
        rbr4 = Conv2D(288, (3, 3), activation="relu", padding="same")(rbr4)
        rbr4 = Conv2D(320, (3, 3), activation="relu", strides=(2, 2))(rbr4)
 
        merged = concatenate([rbr1, rbr2, rbr3, rbr4], axis=-1)
        rbr = BatchNormalization(axis=-1)(merged)
        rbr = Activation("relu")(rbr)
    return rbr
 
## min size: 300 * 300 * 1 
def inception_resnet_v2(img_h, img_w, nb_classes, channels=3, scale=True):
    '''Creates the Inception_ResNet_v1 network.'''
 
    init = Input((img_h, img_w, channels))  # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering
 
    # Input shape is 299 * 299 * 3
    x = resnet_v2_stem(init)  # Output: 35 * 35 * 256
 
    # 5 x Inception A
    for i in range(5):
        x = inception_resnet_v2_A(x, scale_residual=scale)
        # Output: 35 * 35 * 256
 
    # Reduction A
    x = reduction_resnet_A(x, k=256, l=256, m=384, n=384)  # Output: 17 * 17 * 896
 
    # 10 x Inception B
    for i in range(10):
        x = inception_resnet_v2_B(x, scale_residual=scale)
        # Output: 17 * 17 * 896
 
    # Reduction B
    x = reduction_resnet_v2_B(x)  # Output: 8 * 8 * 1792
 
    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_v2_C(x, scale_residual=scale)
        # Output: 8 * 8 * 1792
 
    # Average Pooling
    x = AveragePooling2D((8, 8))(x)  # Output: 1792
 
    # Dropout
    x = Dropout(0.2)(x)  # Keep dropout 0.2 as mentioned in the paper
    x = Flatten()(x)  # Output: 1792
 
    # Output layer
    output = Dense(units=nb_classes, activation="softmax")(x)  # Output: 10000
 
    model = Model(init, output, name="Inception-ResNet-v2")
 
    return model

## way to use inception_resnet_v2
## min-size: 299 * 299
## model = inception_resnet_v2(img_w, img_h, tone_num, channels=1)

# -------------------inception_resnet_v2{E}-------------------



# -------------------inceptionv4{S}-------------------

from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
 
 
CONV_BLOCK_COUNT = 0  # 用来命名计数卷积编号
INCEPTION_A_COUNT = 0
INCEPTION_B_COUNT = 0
INCEPTION_C_COUNT = 0
 
 
def conv_block(x, nb_filters, nb_row, nb_col, strides=(1, 1), padding='same', use_bias=False):
    global CONV_BLOCK_COUNT
    CONV_BLOCK_COUNT += 1
    with K.name_scope('conv_block_'+str(CONV_BLOCK_COUNT)):
        x = Conv2D(filters=nb_filters,
                   kernel_size=(nb_row, nb_col),
                   strides=strides,
                   padding=padding,
                   use_bias=use_bias)(x)
        x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
        x = Activation("relu")(x)
    return x
 
 
def stem(x_input):
    with K.name_scope('stem'):
        x = conv_block(x_input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = conv_block(x, 32, 3, 3, padding='valid')
        x = conv_block(x, 64, 3, 3)
 
        x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
        x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding='valid')
 
        x = concatenate([x1, x2], axis=-1)
 
        x1 = conv_block(x, 64, 1, 1)
        x1 = conv_block(x1, 96, 3, 3, padding='valid')
 
        x2 = conv_block(x, 64, 1, 1)
        x2 = conv_block(x2, 64, 7, 1)
        x2 = conv_block(x2, 64, 1, 7)
        x2 = conv_block(x2, 96, 3, 3, padding='valid')
 
        x = concatenate([x1, x2], axis=-1)
 
        x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding='valid')
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
 
        merged_vector = concatenate([x1, x2], axis=-1)
    return merged_vector
 
 
def inception_A(x_input):
    """35*35 卷积块"""
    global INCEPTION_A_COUNT
    INCEPTION_A_COUNT += 1
    with K.name_scope('inception_A' + str(INCEPTION_A_COUNT)):
        averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)  # 35 * 35 * 192
        averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 96, 1, 1)  # 35 * 35 * 96
 
        conv1x1 = conv_block(x_input, 96, 1, 1)  # 35 * 35 * 96
 
        conv1x1_3x3 = conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
        conv1x1_3x3 = conv_block(conv1x1_3x3, 96, 3, 3)  # 35 * 35 * 96
 
        conv3x3_3x3 = conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
        conv3x3_3x3 = conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96
        conv3x3_3x3 = conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96
 
        merged_vector = concatenate([averagepooling_conv1x1, conv1x1, conv1x1_3x3, conv3x3_3x3], axis=-1)  # 35 * 35 * 384
    return merged_vector
 
 
def inception_B(x_input):
    """17*17 卷积块"""
    global INCEPTION_B_COUNT
    INCEPTION_B_COUNT += 1
    with K.name_scope('inception_B' + str(INCEPTION_B_COUNT)):
        averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
        averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 128, 1, 1)
 
        conv1x1 = conv_block(x_input, 384, 1, 1)
 
        conv1x7_1x7 = conv_block(x_input, 192, 1, 1)
        conv1x7_1x7 = conv_block(conv1x7_1x7, 224, 1, 7)
        conv1x7_1x7 = conv_block(conv1x7_1x7, 256, 1, 7)
 
        conv2_1x7_7x1 = conv_block(x_input, 192, 1, 1)
        conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 192, 1, 7)
        conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 224, 7, 1)
        conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 224, 1, 7)
        conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 256, 7, 1)
 
        merged_vector = concatenate([averagepooling_conv1x1, conv1x1, conv1x7_1x7, conv2_1x7_7x1], axis=-1)
    return merged_vector
 
 
def inception_C(x_input):
    """8*8 卷积块"""
    global INCEPTION_C_COUNT
    INCEPTION_C_COUNT += 1
    with K.name_scope('Inception_C' + str(INCEPTION_C_COUNT)):
        averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
        averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 256, 1, 1)
 
        conv1x1 = conv_block(x_input, 256, 1, 1)
 
        # 用 1x3 和 3x1 替代 3x3
        conv3x3_1x1 = conv_block(x_input, 384, 1, 1)
        conv3x3_1 = conv_block(conv3x3_1x1, 256, 1, 3)
        conv3x3_2 = conv_block(conv3x3_1x1, 256, 3, 1)
 
        conv2_3x3_1x1 = conv_block(x_input, 384, 1, 1)
        conv2_3x3_1x1 = conv_block(conv2_3x3_1x1, 448, 1, 3)
        conv2_3x3_1x1 = conv_block(conv2_3x3_1x1, 512, 3, 1)
        conv2_3x3_1x1_1 = conv_block(conv2_3x3_1x1, 256, 3, 1)
        conv2_3x3_1x1_2 = conv_block(conv2_3x3_1x1, 256, 1, 3)
 
        merged_vector = concatenate([averagepooling_conv1x1, conv1x1, conv3x3_1, conv3x3_2, conv2_3x3_1x1_1, conv2_3x3_1x1_2], axis=-1)
    return merged_vector
 
 
def reduction_A(x_input, k=192, l=224, m=256, n=384):
    with K.name_scope('Reduction_A'):
        """Architecture of a 35 * 35 to 17 * 17 Reduction_A block."""
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)
 
        conv3x3 = conv_block(x_input, n, 3, 3, strides=(2, 2), padding='valid')
 
        conv2_3x3 = conv_block(x_input, k, 1, 1)
        conv2_3x3 = conv_block(conv2_3x3, l, 3, 3)
        conv2_3x3 = conv_block(conv2_3x3, m, 3, 3, strides=(2, 2), padding='valid')
 
        merged_vector = concatenate([maxpool, conv3x3, conv2_3x3], axis=-1)
    return merged_vector
 
 
def reduction_B(x_input):
    """Architecture of a 17 * 17 to 8 * 8 Reduction_B block."""
    with K.name_scope('Reduction_B'):
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)
 
        conv3x3 = conv_block(x_input, 192, 1, 1)
        conv3x3 = conv_block(conv3x3, 192, 3, 3, strides=(2, 2), padding='valid')
 
        conv1x7_7x1_3x3 = conv_block(x_input, 256, 1, 1)
        conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 256, 1, 7)
        conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 320, 7, 1)
        conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 320, 3, 3, strides=(2, 2), padding='valid')
 
        merged_vector = concatenate([maxpool, conv3x3, conv1x7_7x1_3x3], axis=-1)
    return merged_vector
 
 
def inception_v4(img_w, img_h, tone_num, channels=3):
    x_input = Input(shape=(img_w, img_h, channels))
    # Stem
    x = stem(x_input)  # 35 x 35 x 384
    # 4 x Inception_A
    for i in range(4):
        x = inception_A(x)  # 35 x 35 x 384
    # Reduction_A
    x = reduction_A(x, k=192, l=224, m=256, n=384)  # 17 x 17 x 1024
    # 7 x Inception_B
    for i in range(7):
        x = inception_B(x)  # 17 x 17 x1024
    # Reduction_B
    x = reduction_B(x)  # 8 x 8 x 1536
    # Average Pooling
    x = AveragePooling2D(pool_size=(8, 8))(x)  # 1536
    # dropout
    x = Dropout(0.2)(x)
    x = Flatten()(x)  # 1536
    # 全连接层
    x = Dense(units=tone_num, activation='softmax')(x)
    model = Model(inputs=x_input, outputs=x, name='Inception-V4')
    return model


## way to use inceptionv4
## min-size: 299 * 299
## model = inceptionv4(img_w, img_h, tone_num, channels=1)

# -------------------inceptionv4{E}-------------------    



# include imagenet top, transfer learning build model
# inceptionv3, vgg16, vgg19, resnet50, mobilenet, Xception, densenet121, densenet169, densenet201, inceptionresnetv2
def build_model_options(application, img_h, img_w , num_class, channels=3):
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
    
    INPUTSIZE = (img_h, img=w, channels)

    # min size: 139 * 139 * 1  => voice mfcc  
    if application == 'inceptionv3':
        base_model = InceptionV3(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')          
    # smallest: 48 * 48 default: 224 * 224    
    elif application == 'vgg16':
        base_model = VGG16(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')  
    elif application == 'vgg19':
        base_model = VGG19(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')  
    # smallest: 197 * 197 default: 224 * 224      
    elif application == 'resnet50':
        base_model = ResNet50(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')  
    # smallest: 32 * 32 default: 224 * 224    
    elif application == 'mobilenet':
        base_model = MobileNet(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')  
    elif application == 'Xception':
        base_model = Xception(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')  
    elif application == 'densenet121':
        base_model = densenet.DenseNet121(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')  
    elif application == 'densenet169':
        base_model = densenet.DenseNet169(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')
    # min size: 221 * 221 * 1  => voice mfcc
    elif application == 'densenet201':       
        print('densenet201')
        base_model = densenet.DenseNet201(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')        
    elif application == 'nasnetlarge':
        base_model = densenet.NASNetLarge(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')
    elif application == 'nasnetmobile':
        base_model = densenet.NASNetMobile(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')
    elif application == 'inceptionresnetv2':
        base_model = InceptionResNetV2(include_top=False, weights=None,
            input_shape=INPUTSIZE, pooling='max')
    
    x = base_model.output
    #x = Dense(lastdensesize, activation='relu')(x)
    # 最後一層不要 batch normalize / dropout
    outputs = Dense(num_class, activation='softmax')(x)    
    model = Model(base_model.inputs, outputs)
    #model.summary()
    
    return model