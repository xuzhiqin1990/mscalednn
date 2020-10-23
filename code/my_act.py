# !python3
# coding: utf-8
# author: Ziqi Liu and Zhi-Qin John Xu
# Reference: Ziqi Liu，Wei Cai，Zhi-Qin John Xu. Multi-scale Deep Neural Network (MscaleDNN)
# for Solving Poisson-Boltzmann Equation in Complex Domains[J]. 2020. arXiv:2007.11207 (CiCP)

import tensorflow as tf
import numpy as np


def relu2(x):
    return tf.nn.relu(x) ** 2


def relu3(x):
    return tf.nn.relu(x) ** 3


def srelu(x):
    return tf.nn.relu(1 - x) * tf.nn.relu(x)


def wave(x):
    return (
        tf.nn.relu(x)
        - 2 * tf.nn.relu(x - 1 / 4)
        + 2 * tf.nn.relu(x - 3 / 4)
        - tf.nn.relu(x - 1)
    )


def ex2(x):
    return tf.exp(-tf.square(x))


def xex2(x):
    return x * tf.exp(-tf.square(x))


def phi2(x):
    return tf.nn.relu(x) - 2 * tf.nn.relu(x - 1) + tf.nn.relu(x - 2)


def psi2(x):
    return (
        1 / 12 * tf.nn.relu(2 * x)
        - 2 / 3 * tf.nn.relu(2 * x - 1)
        + 23 / 12 * tf.nn.relu(2 * x - 2)
        - 8 / 3 * tf.nn.relu(2 * x - 3)
        + 23 / 12 * tf.nn.relu(2 * x - 4)
        - 2 / 3 * tf.nn.relu(2 * x - 5)
        + 1 / 12 * tf.nn.relu(2 * x - 6)
    )


def phi3(x):
    return (
        1 / 2 * tf.nn.relu(x - 0) ** 2
        - 3 / 2 * tf.nn.relu(x - 1) ** 2
        + 3 / 2 * tf.nn.relu(x - 2) ** 2
        - 1 / 2 * tf.nn.relu(x - 3) ** 2
    )


def psi3(x):
    return (
        1 / 960 * tf.nn.relu(2 * x - 0) ** 2
        - 32 / 960 * tf.nn.relu(2 * x - 1) ** 2
        + 237 / 960 * tf.nn.relu(2 * x - 2) ** 2
        - 832 / 960 * tf.nn.relu(2 * x - 3) ** 2
        + 1682 / 960 * tf.nn.relu(2 * x - 4) ** 2
        - 2112 / 960 * tf.nn.relu(2 * x - 5) ** 2
        + 1682 / 960 * tf.nn.relu(2 * x - 6) ** 2
        - 832 / 960 * tf.nn.relu(2 * x - 7) ** 2
        + 237 / 960 * tf.nn.relu(2 * x - 8) ** 2
        - 32 / 960 * tf.nn.relu(2 * x - 9) ** 2
        + 1 / 960 * tf.nn.relu(2 * x - 10) ** 2
    )

def s2relu(x):
    return (
        tf.sin(2*np.pi*x)*srelu(x)
    )


act_dict = {
    "relu": tf.nn.relu,
    "relu2": relu2,
    "relu3": relu3,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh,
    "softplus": tf.nn.softplus,
    "sin": tf.sin,
    "cos": tf.cos,
    "ex2": ex2,
    "xex2": xex2,
    "srelu": srelu,
    "wave": wave,
    "phi2": phi2,
    "psi2": psi2,
    "phi3": phi3,
    "psi3": psi3,
    "s2relu": s2relu,
}
