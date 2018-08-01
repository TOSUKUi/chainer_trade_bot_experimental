import chainer
from chainer.datasets import mnist
import matplotlib.pyplot as plt
from chainer.datasets import split_dataset_random
from chainer import iterators
import chainer.links as L
import chainer.functions as F
from chainer.cuda import to_cpu
import argparse
import numpy
import random

def reset_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


class MLP(chainer.Chain):
    
    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, None)

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')


    reset_seed(0)


    train_val, test = mnist.get_mnist(withlabel=True, ndim=1)
    train, valid = split_dataset_random(train_val, 50000, seed=0)

    print('Training dataset size:', len(train))
    print('Validation dataset size:', len(valid))

    batchsize = 128

    train_iter = iterators.SerialIterator(train, batchsize)
    valid_iter = iterators.SerialIterator(valid, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize)

    net = MLP(n_mid_units=args.unit)

    optimizer = chainer.optimizers.SGD(lr=0.01).setup(net)

    # 学習
    gpu_id = args.gpu

    if gpu_id >= 0:
        net.to_gpu(gpu_id)
    while train_iter.epoch < args.epoch:
        #-------------------------------
        train_batch = train_iter.next()
        x, t = chainer.dataset.concat_examples(train_batch, args.gpu)
        y = net(x)
        
        loss = F.softmax_cross_entropy(y, t)
        net.cleargrads()
        loss.backward()

        optimizer.update()
        #-----------------------------------

        # 1エポックごとにValidationデータに対する予測精度を図ってモデルの汎化性能が向上していることをチェックしよう
        if train_iter.is_new_epoch:
            print("epoch:{:02d}train_loss:{:.04f}".format(
                train_iter.epoch, float(to_cpu(loss.data))
            ), end='')

        valid_losses = []
        valid_accuracies = []

        while True:
            valid_batch = valid_iter.next()
            x_valid, t_valid = chainer.dataset.concat_examples(valid_batch, gpu_id)
        
        # validationデータをforward
            with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
                y_valid = net(x_valid)
        
            # lossを計算
            loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
            valid_losses.append(to_cpu(loss_valid.array))


            # 精度を計算
            accuracy = F.accuracy(y_valid, t_valid)
            accuracy.to_cpu()
            valid_accuracies.append(accuracy.array)

            if valid_iter.is_new_epoch:
                valid_iter.reset()
                break
        
        print("val_loss:{:.04f} val_accuracy:{:.04f}".format(
            numpy.mean(valid_losses), numpy.mean(valid_accuracies)
        ))


    # テストデータでの評価
    test_accuracies = []

    while True:
        test_batch = test_iter.next()
        x_test, t_test = chainer.dataset.concat_examples(test_batch, gpu_id)

        with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
            y_test = net(x_test)
        
        # 精度を計算
        accuracy = F.accuracy(y_test, t_test)
        accuracy.to_cpu()
        test_accuracies.append(accuracy.array)

        if test_iter.is_new_epoch:
            test_iter.reset()
            break


    print("test_accuracy:{:.04f}".format(numpy.mean(test_accuracies)))
        

main()


        


