from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers import Input,add
from keras.layers.core import Activation
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def get_unet(patch_height, patch_width, channels, n_classes):
    """
    It creates a U-Net and returns the model
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the model (unet)
    """
    axis = 3
    k = 3 # kernel size
    s = 2 # stride
    n_filters = 32 # number of filters

    #初始化keras张量
    inputs = Input((patch_height, patch_width, channels))

    # n_filters：输出的维度 （k，k）：卷积核尺寸 padding：边缘填充
    # 400,400,3 ==> 400,400,32
    conv1 = Conv2D(n_filters, (k,k), padding='same')(inputs)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k), padding='same')(conv1)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    # 400,400,32 ==> 200,200,32
    pool1 = MaxPooling2D(pool_size=(s,s))(conv1)

    # 200,200,32 ==> 200,200,64
    conv2 = Conv2D(2*n_filters, (k,k), padding='same')(pool1)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding='same')(conv2)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    # 200,200,64 ==> 100,100,64
    pool2 = MaxPooling2D(pool_size=(s,s))(conv2)

    # 100,100,64 ==> 100,100,128
    conv3 = Conv2D(4*n_filters, (k,k), padding='same')(pool2)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding='same')(conv3)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    # 100,100,128 ==> 50,50,128
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    # 50,50,128 ==> 50,50,256
    conv4 = Conv2D(8 * n_filters, (k, k), padding='same')(pool3)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    # 50,50,256 ==> 25,25,256
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    # 25,25,256 ==> 25,25,512
    conv5 = Conv2D(16 * n_filters, (k, k), padding='same')(pool4)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)

    #先上采样放大 在进行卷积操作 相当于转置卷积 并进行 拼接
    # 25,25,512 ==> 50,50,768
    up1 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8 * n_filters, (k,k), padding='same')(up1)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding='same')(conv6)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)

    #50,50,768 ==> 100,100,896
    up2 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(up2)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(conv7)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)

    # 100,100,896 ==> 200,200,960
    up3 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(up3)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(conv8)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)

    # 200,200,960 ==> 400,400,992
    up4 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding='same')(up4)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding='same')(conv9)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)

    # 全连接层 400,400,992 ==> 400,400,5
    outputs = Conv2D(n_classes, (1,1), padding='same', activation='softmax')(conv9)

    unet = Model(inputs=inputs, outputs=outputs)

    return unet


def get_unet1(patch_height, patch_width, channels, n_classes):
    """
    It creates a U-Net and returns the model and backbone is ResNet
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the model (vggnet)
    """
    axis = 3
    k = 3 # kernel size
    s = 2 # stride
    n_filters = 32 # number of filters 通道数

    #初始化keras张量
    inputs = Input((patch_height, patch_width, channels))

    # n_filters：输出的维度 （k，k）：卷积核尺寸 padding：边缘填充
    # 400,400,3 ==> 400,400,32
    conv1 = Conv2D(n_filters, (k, k), padding='same')(inputs)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k), padding='same')(conv1)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    # 400,400,32 ==> 200,200,32
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    # 200,200,32 ==> 200,200,32
    conv2 = Conv2D(n_filters, (k, k), padding='same')(pool1)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(n_filters, (k, k), padding='same')(conv2)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = add([pool1, conv2])
    conv2 = Activation('relu')(conv2)
    x2 = Conv2D(2 * n_filters, (k, k), padding='same')(conv2)
    # 200,200,32 ==> 100,100,64
    output2 = Conv2D(n_filters * 2, (1,1),padding='same',strides=s)(conv2)

    # 200,200,32 ==> 100,100,64
    conv3 = Conv2D(2 * n_filters, (k, k), padding='same',strides=s)(conv2)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    # 100,100,64 ==> 100,100,64
    conv3 = Conv2D(2 * n_filters, (k, k), padding='same')(conv3)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = add([output2,conv3])
    conv3 = Activation('relu')(conv3)
    x3 = Conv2D(4 * n_filters, (k, k), padding='same')(conv3)
    # 100,100,64 ==> 50,50,128
    output3 = Conv2D(n_filters * 4,(1,1),padding='same',strides=s)(conv3)

    # 100,100,64 ==> 50,50,128
    conv4 = Conv2D(4 * n_filters, (k, k), padding='same',strides=s)(conv3)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(4 * n_filters, (k, k), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = add([output3,conv4])
    conv4 = Activation('relu')(conv4)
    # x4 == 50,50,256
    x4 = Conv2D(8 * n_filters,(k,k),padding='same')(conv4)
    # 50,50,128 ==> 25,25,256
    output4 = Conv2D(8 * n_filters,(1,1),padding='same',strides=s)(conv4)

    # 50,50,128 ==> 25,25,256
    conv5 = Conv2D(8 * n_filters, (k, k), padding='same',strides=s)(conv4)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(8 * n_filters, (k, k), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = add([output4,conv5])
    conv5 = Activation('relu')(conv5)
    # 25,25,256 ==> 25,25,512
    output5 = Conv2D(16 * n_filters,(1,1),padding='same')(conv5)

    # 先上采样放大 在进行卷积操作 相当于转置卷积 并进行拼接
    # 25,25,512 ==> 50,50,768
    up1 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(output5), x4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding='same')(up1)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding='same')(conv6)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)

    # 50,50,768 ==> 100,100,896
    up2 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv6), x3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(up2)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(conv7)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)

    # 100,100,896 ==> 200,200,960
    up3 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv7), x2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(up3)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(conv8)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)

    # 200,200,960 ==> 400,400,992
    up4 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding='same')(up4)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding='same')(conv9)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)

    # 全连接层 400,400,992 ==> 400,400,5
    outputs = Conv2D(n_classes, (1, 1), padding='same', activation='softmax')(conv9)

    unet = Model(inputs=inputs, outputs=outputs)
    return unet


def get_unet2(patch_height, patch_width, channels, n_classes):
    """
    It creates a U-Net and returns the model and backbone is DenseNet
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the model (unet)
    """
    axis = 3
    k = 3 # kernel size 卷积核大小
    s = 2 # stride   步长
    n_filters = 32 # number of filters  通道数

    #初始化keras张量
    inputs = Input((patch_height, patch_width, channels))

    # n_filters：输出的维度 （k，k）：卷积核尺寸 padding：边缘填充
    # 400,400,3 ==> 400,400,32
    conv1 = Conv2D(n_filters, (k,k), padding='same')(inputs)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k), padding='same')(conv1)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    # 400,400,32 ==> 200,200,32
    pool1 = MaxPooling2D(pool_size=(s,s))(conv1)

    # DenseBlock模块 200,200,32 ==> 200,200,32
    conv2 = Conv2D(n_filters, (1,1), padding='same')(pool1)
    # battleneck层 第一层Dense
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv2)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(n_filters,(k,k),padding='same')(conv2)
    # 200,200,32 ==> 200,200,64
    x1 = Concatenate(axis=axis)([conv2,pool1])

    # Transition层
    ts = Conv2D(n_filters * 4, (1, 1), padding='same')(x1)
    # 200,200,64 ==> 100,100,64  #将pool2 看成 x1
    pool2 = AveragePooling2D(pool_size=(s, s), strides=2)(ts)

    conv3 = Conv2D(n_filters,(1,1),padding='same')(pool2)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv3)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(n_filters, (k, k), padding='same')(conv3)
    # 100,100,32 ==> 100,100,96
    tmp2 = Concatenate(axis=axis)([conv3,pool2])

    conv3 = Conv2D(n_filters, (1, 1), padding='same')(tmp2)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv3)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(n_filters, (k, k), padding='same')(conv3)
    # 100,100,32 ==> 100,100,128
    x2 = Concatenate(axis=axis)([conv3, tmp2])

    # Transition层
    ts1 = Conv2D(n_filters * 4,(1,1),padding='same')(x2)
    # 100,100,128 ==> 50,50,128
    pool2 = AveragePooling2D(pool_size=(s,s),strides=2)(ts1)

    # 50,50,128 ==> 50,50,32
    conv4 = Conv2D(n_filters, (1, 1), padding='same')(pool2)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(n_filters, (k, k), padding='same')(conv4)
    # 50,50,32 ==> 50,50,160
    tmp3 = Concatenate(axis=axis)([conv4,pool2])

    # 50,50,160 ==> 50,50,32
    conv4 = Conv2D(n_filters, (1, 1), padding='same')(tmp3)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(n_filters, (k, k), padding='same')(conv4)
    # 50,50,32 ==> 50,50,192
    tmp4 = Concatenate(axis=axis)([conv4, tmp3])

    # 50,50,192 ==> 50,50,32
    conv4 = Conv2D(n_filters, (1, 1), padding='same')(tmp4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(n_filters, (k, k), padding='same')(conv4)
    # 50,50,32 ==> 50,50,224
    tmp5 = Concatenate(axis=axis)([conv4, tmp4])

    # 50,50,224 ==> 50,50,32
    conv4 = Conv2D(n_filters, (1, 1), padding='same')(tmp5)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(n_filters, (k, k), padding='same')(conv4)
    # 50,50,32 ==> 50,50,256
    x3 = Concatenate(axis=axis)([conv4, tmp5])

    ts2 = Conv2D(n_filters * 8, (1, 1), padding='same')(x3)
    # 50,50,256 ==> 25,25,256
    pool3 = AveragePooling2D(pool_size=(s, s), strides=2)(ts2)

    # 25,25,256 ==> 25,25,32
    conv5 = Conv2D(n_filters, (1, 1), padding='same')(pool3)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(n_filters, (k, k), padding='same')(conv5)
    # 25,25,32 ==> 25,25,288
    tmp6 = Concatenate(axis=axis)([conv5, pool3])

    # 50,50,288 ==> 50,50,32
    conv5 = Conv2D(n_filters, (1, 1), padding='same')(tmp6)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(n_filters, (k, k), padding='same')(conv5)
    # 50,50,32 ==> 50,50,320
    tmp7 = Concatenate(axis=axis)([conv5, tmp6])

    # 50,50,288 ==> 50,50,32
    conv5 = Conv2D(n_filters, (1, 1), padding='same')(tmp7)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(n_filters, (k, k), padding='same')(conv5)
    # 50,50,32 ==> 50,50,352
    tmp8 = Concatenate(axis=axis)([conv5, tmp7])

    # 50,50,352 ==> 50,50,32
    conv5 = Conv2D(n_filters, (1, 1), padding='same')(tmp8)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(n_filters, (k, k), padding='same')(conv5)
    # 50,50,32 ==> 50,50,384
    tmp9 = Concatenate(axis=axis)([conv5, tmp8])

    # 50,50,352 ==> 50,50,32
    conv5 = Conv2D(n_filters, (1, 1), padding='same')(tmp9)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(n_filters, (k, k), padding='same')(conv5)
    # 50,50,32 ==> 50,50,416
    tmp10 = Concatenate(axis=axis)([conv5, tmp9])

    # 50,50,352 ==> 50,50,32
    conv5 = Conv2D(n_filters, (1, 1), padding='same')(tmp10)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(n_filters, (k, k), padding='same')(conv5)
    # 50,50,32 ==> 50,50,448
    tmp11 = Concatenate(axis=axis)([conv5, tmp10])

    conv5 = Conv2D(n_filters, (1, 1), padding='same')(tmp11)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(n_filters, (k, k), padding='same')(conv5)
    # 50,50,32 ==> 50,50,480
    tmp12 = Concatenate(axis=axis)([conv5, tmp11])

    conv5 = Conv2D(n_filters, (1, 1), padding='same')(tmp12)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(4 * n_filters, (1, 1), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(n_filters, (k, k), padding='same')(conv5)
    # 25,25,32 ==> 25,25,512
    conv5 = Concatenate(axis=axis)([conv5, tmp12])

    #先上采样放大 在进行卷积操作 相当于转置卷积 并进行拼接
    # 25,25,512 ==> 50,50,768
    up1 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv5), x3])
    conv6 = Conv2D(8 * n_filters, (k,k), padding='same')(up1)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding='same')(conv6)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)

    #50,50,768 ==> 100,100,896
    up2 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv6), x2])
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(up2)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(conv7)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)

    # 100,100,896 ==> 200,200,960
    up3 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv7), x1])
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(up3)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(conv8)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)

    # 200,200,960 ==> 400,400,992
    up4 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding='same')(up4)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding='same')(conv9)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)

    # 全连接层 400,400,992 ==> 400,400,5
    outputs = Conv2D(n_classes, (1,1), padding='same', activation='softmax')(conv9)

    unet = Model(inputs=inputs, outputs=outputs)

    return unet