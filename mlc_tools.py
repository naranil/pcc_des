import numpy as np

"""
Losses
"""
def true_loss(W, Z):
    if len(W.shape) == 1:
        W = W.reshape((1, -1))
        Z = Z.reshape((1, -1))
    return sum(map(lambda zw: int(sum(zw[1][zw[0].astype(bool)]) / sum(zw[0]) <= 0.5) if sum(zw[0]) != 0 else 1, \
                   zip(Z, W)))

def precision_loss(W, Z):
    if len(W.shape) == 1:
        W = W.reshape((1, -1))
        Z = Z.reshape((1, -1))
    return sum(map(lambda zw: 1-sum(zw[1][zw[0].astype(bool)]) / sum(zw[0]) if sum(zw[0]) != 0 else 1, \
                   zip(Z, W))) / W.shape[0]