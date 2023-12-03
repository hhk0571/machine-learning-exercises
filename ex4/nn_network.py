#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split

def oneHot(y, k=10):
    return np.eye(k)[y[:, 0]]

class DataLoader(object):
    def __init__(self, fmt='mat'):
        self.fmt = fmt

    def getTrainingData(self, filename: str):
        '''
        return X_train, X_test, y_train, y_test
        '''
        if self.fmt == 'mat':
            data = loadmat(filename)
            return train_test_split(data['X'], data['y']-1, test_size=0.3)
        else:
            print(f'unsupported format: {self.fmt}')

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidGradient(z):
    '''
    sigmoid 的导数
    '''
    return sigmoid(z) * (1 - sigmoid(z))

class NN_Network(object):
    def __init__(self, sizes: list, lamda=1):
        self.sizes = sizes
        self.layerNum = len(sizes)
        self.thetas = [np.zeros((currSize, prevSize + 1))
                       for currSize, prevSize in zip(sizes[1:], sizes[:-1])]

    @staticmethod
    def serialize(thetas: list) -> np.ndarray:
        flatten_thetas = [x.flatten() for x in thetas]
        return np.concatenate(flatten_thetas, axis=0)

    def deserialize(self, serializedThetas) -> list:
        thetas = []
        _start = 0
        for theta in self.thetas:
            flatten_theta = serializedThetas[_start:_start + theta.size]
            thetas.append(flatten_theta.reshape(theta.shape))
            _start = _start + theta.size
        return thetas

    def updateThetas(self, thetas: list):
        self.thetas = thetas

    def feedForward(self, X: np.ndarray):
        a_values = []
        z_values = []
        a = X
        for theta in self.thetas:
            a = np.insert(a, 0, values=1, axis=1) # 给bias 增加一列 1
            a_values.append(a)
            z = a @ theta.T
            a = sigmoid(z)
            z_values.append(z)
        a_values.append(a) # 输出层(最后一层)的a值
        return z_values, a_values


    def computeCost(self, serializeThetas, X, y, lamda=1):
        '''
        计算损失和梯度, 给 scipy.optimize.fmin_tnc 调用

        serializeThetas 参数由 fmin_tnc 传入
        '''
        self.thetas = self.deserialize(serializeThetas)
        m = len(y) # number of training examples
        z_values, a_values = self.feedForward(X)

        ## 计算损失
        h = a_values[-1]
        J = -1/m*np.sum(y * np.log(h) + (1-y)*np.log(1-h))
        sum = 0.0
        for theta in self.thetas:
            sum += np.sum(np.power(theta[1:], 2))
        reg = lamda/(2*m) * sum
        J += reg


        ## 计算梯度
        deltazs = []
        deltazs.append(h - y) # the last layer
        for theta, z in zip(reversed(self.thetas), reversed(z_values[:-1])):
            deltazs.append( deltazs[-1] @ theta[:, 1:] * sigmoidGradient(z))

        grads = []
        for deltaz, a in zip(reversed(deltazs), a_values):
            grads.append(deltaz.T @ a / m)

        ## regularize
        for grad, theta in zip(grads, self.thetas):
            grad[1:] += lamda / m * theta[1:]

        return J, self.serialize(grads)


    def train(self, X: np.ndarray, y: np.ndarray, maxIter=200, lamda=1):
        from scipy.optimize import fmin_tnc
        import time
        _start = time.time()
        init_theta = self.serialize(self.thetas)
        init_theta = np.random.uniform(-0.5, 0.5, init_theta.size)
        print('training ongoing...')
        result = fmin_tnc(func = self.computeCost,
                          x0 = init_theta,
                          maxfun = maxIter,
                          args = (X, y, lamda))
        _stop = time.time()
        print(result)
        print('\ntraining completed.')
        print(f'\nElapsed time: {_stop - _start:.1f}s\n')

        self.thetas = self.deserialize(result[0])


    def predict(self, X: np.ndarray):
        __, a_values = self.feedForward(X)
        h = a_values[-1]
        h_argmax = np.argmax(h, axis=1) # 按行返回概率最大的索引, 对应数字0 ~ 9
        return h_argmax[:, np.newaxis] # 维度变为 (5000,1)

    def saveThetas2Mat(self, filename):
        mdic = {"label": "experiment"}
        for i, theta in enumerate(self.thetas, 1):
            mdic[f'Theta{i}'] = theta
        savemat(filename, mdic)
        print('save mat file:', filename)


def test_ready_trained_thetas(X_train, y_train):
    weights = loadmat('ex4weights.mat')
    print(weights.keys())

    Theta1 = weights['Theta1'] # 25 * 401 第1层的权重 (中的隐藏层 25个神经元)
    Theta2 = weights['Theta2'] # 10 * 26  第2层的权重 (输出层 10个神经元)

    print(Theta1.shape, Theta2.shape) # (25, 401) (10, 26)

    nn = NN_Network([400, 25, 10])
    nn.updateThetas([Theta1, Theta2])
    y_pred = nn.predict(X_train)

    print(y_train.flatten())
    print(y_pred.flatten())

    acc = np.mean(y_pred == y_train) # 不能直接与y比较, 因为y已经经过oneHot转换

    print(f'accuracy with ready-trained weigths: {acc:.3f}') # 0.975 左右


def plot_hidden_layer(theta):
    hidden_layer = theta[:, 1:]  # (25, 400)

    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(8, 8), sharex=True, sharey=True)

    for r in range(5):
        for c in range(5):
            ax[r, c].imshow(hidden_layer[5 * r + c].reshape(20, 20).T, cmap='gray_r')

    plt.xticks([])
    plt.yticks([])

    plt.show()


def main():
    dataloader = DataLoader()
    X_train, X_test, y_train, y_test = dataloader.getTrainingData('ex4data1.mat')
    print(X_train.shape, y_train.shape)

    test_ready_trained_thetas(X_train, y_train)

    nn = NN_Network([400, 100, 10]) # [400, 25, 10] is enough
    nn.train(X_train, oneHot(y_train), maxIter=500, lamda=1.5)

    y_pred = nn.predict(X_train)
    acc = np.mean(y_pred == y_train)
    print(f'accuracy with X_train: {acc:.3f}') # ~0.992

    y_pred = nn.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f'accuracy with X_test : {acc:.3f}') # ~0.931

    nn.saveThetas2Mat(f'{acc:.3f}_thetas.mat')

    # plot_hidden_layer(nn.thetas[0])



if __name__ == "__main__":
    sys.exit(main())
