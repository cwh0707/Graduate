import os
from os.path import join
from model_unet import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from unet_generator import UNetGeneratorClass
from utils.losses import *
from utils.params import parse_arguments_unet, default_params
import sys

path_to_data = '../'

def train_unet_generator(**params):
    """
    It trains the U-Net first using patches with a percentage (thres_score) of lesion and after using patches with lesion.
    It saves the best weights using model checkpointer
    :param data_path: path where images and labels are located
    :param out_path: path where the evolution of the performance (TensorBorad) is saved
    """

    params = dict(
        default_params,
        **params
    )
    verbose = params['verbose']

    if verbose:
        print("Welcome to U-Net training")

    # tensorboard保存的就是对应的文件夹
    out_path = join(path_to_data, params['weights_path'] + 'DenseNettensorboard/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        if os.listdir(out_path):
            os.system('rm ' + out_path + '*')

    if verbose:
        print("Getting model...")

    patch_size = params['patch_size']
    channels = params['channels']
    n_classes = params['n_classes']

    # 返回一个unet模型
    unet = get_unet(patch_size, patch_size, channels, n_classes)

    # 得到网络的网络结构
    unet.summary()

    # 计算损失效益
    metrics = [generalised_dice_coef]
    lr = params['lr']
    # 注意 这里的dc值取得是5类别的平均值
    loss = gen_dice_multilabel

    #先编译
    # optimizer：优化器 使用的是Adam 学习率是lr
    # loss： 损失函数
    # merics： 评估模型在训练和测试时的性能的指标
    unet.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)

    data_path = join(path_to_data, params['data_path'])
    batch_size = params['batch_size']

    if verbose:
        print("Getting generators...")

    #生成生成器 可以pass
    train_scored_generator = UNetGeneratorClass(data_path=data_path, n_class=n_classes, batch_size=batch_size,
                                                channels=channels, apply_augmentation=False, thres_score=0.3, train=True)
    #测试集生成器 pass
    train_generator = UNetGeneratorClass(data_path=data_path, n_class=n_classes, batch_size=batch_size,
                                         channels=channels, apply_augmentation=True, thres_score=0, train=True)
    # 可以pass
    val_generator = UNetGeneratorClass(data_path=data_path, n_class=n_classes, batch_size=batch_size,
                                       channels=channels, apply_augmentation=False, thres_score=None, train=False)

    if verbose:
        print("Files in scored generator for training:", len(train_scored_generator.files))
        print("Training model...")

    weights_path = join(path_to_data, params['weights_path'])
    # best_weights 中存的是输入的 weights参数 后面加上.h5文件路径
    best_weights = join(weights_path, params['weights'] + '.h5')

    #保存模型的一些信息 moitor：需要监视的值 save_best_only：保存性能最好的模型
    # param1：文件路径/url
    model_checkpoint = ModelCheckpoint(best_weights, verbose=1, monitor='val_loss', save_best_only=True)
    tensorboard = TensorBoard(log_dir=out_path, histogram_freq=0, write_graph=True, write_images=False)

    #训练模型
    # param1：gengrator生成器（image，mask）
    # param2：训练steps_per_epoch个数据记一个epoc结束
    # param3：数据迭代轮数 verbose：信息展示模式
    # callbacks： 回调函数

    unet.fit_generator(train_scored_generator.generate(),
                       steps_per_epoch=(len(train_scored_generator.files) // train_scored_generator.batch_size + 1),
                       epochs=13, verbose=verbose)

    unet.fit_generator(train_generator.generate(),
                       steps_per_epoch=(len(train_generator.files) // train_generator.batch_size + 1),
                       epochs=30, verbose=verbose, callbacks=[tensorboard, model_checkpoint],
                       validation_data=val_generator.generate(),
                       validation_steps=(len(val_generator.files) // val_generator.batch_size + 1))


    # unet.save_weights(join(weights_path, 'last_weights.h5'), overwrite=True)

    if verbose:
        print("Training finished")


if __name__ == '__main__':
    train_unet_generator(**vars(parse_arguments_unet()))