
"""
 Yusuf Alptug Aydin
 260201065
"""

import torch
import numpy as np


class UtilsProvider:
    def __init__(self):
        pass

    def activation(self, f_type, x):
        if (f_type == "tanh"):
            return self.tanh_act(x)
        if (f_type == "sigmoid"):
            return self.sigmoid_act(x)
        if (f_type == "relu"):
            return self.relu_act(x)
        if (f_type == "softmax"):
            return self.softmax_act(x)

    def der_activation(self, f_type, x):
        if (f_type == "tanh"):
            return self.der_tanh(x)
        if (f_type == "sigmoid"):
            return self.der_sigmoid(x)
        if (f_type == "relu"):
            return self.der_relu(x)

    # tanh --> f(x) = (2 / (1 + e**(-2*x)) ) - 1
    def tanh_act(self, x):
        f = (2 / (1 + torch.exp(x.mul(-2)))) - 1 # f = x.tanh()
        return f

    # tanh prime --> f'(x) = 1 - tanh**2(x)
    def der_tanh(self, x):
        df = 1 - self.tanh_act(x).pow(2)
        return df

    # σ(x)= 1 / (1 + e^(-x))
    def sigmoid_act(self, x):
        f = 1 / (1 + torch.exp(-x))
        return f

    # σ'(x)= σ(x) * (1 - σ(x))
    def der_sigmoid(self, x):
        df = self.sigmoid_act(x) * (1 - self.sigmoid_act(x))
        return df

    # RELU
    def relu_act(self, x):
        return np.maximum(0, x)  # torch.max # Loss için ama yine de https://stackoverflow.com/a/55546848/12579069

    def der_relu(self, x):
        result = x.clone()
        result[x < 0] = 0
        return result

    # Softmax
    def softmax_act(self, x):
        expo = torch.exp(x - x.max())
        return expo / torch.sum(expo)

    def der_softmax(self, x):
        temp = x.reshape(-1, 1)
        return np.diagflat(temp) - np.dot(temp, temp.T)

    # MSE
    def mean_squared_error(self, v, t):
        return (v - t).pow(2).sum().mean()

    # MSE prime = 2 * (y - ypred)
    def der_mse(self, v, t):
        return (v - t).mul(2)

    # CROSS ENTROPY
    # From Nielsen's book:
    # CE = -1/n * ∑ [y * lna + (1−y) * ln(1−a)]
    def cross_entropy_cost(self, v, t):
        total = torch.sum((t * torch.log(v) + (1 - t) * torch.log(1 - v)))
        return -1 * total / t.size(0)

