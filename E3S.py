# !python3
# coding: utf-8
# author: sis-flag

import os
import time
import shutil
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from my_act import act_dict

# %% parameters
R = {}

R["seed"] = 0

R["size_it"] = 6000
R["size_bd"] = 3000

R["penalty"] = 1e3

R["scale"] = np.array([1, 2, 4, 8, 16])

R["act_name"] = ("phi3",) * 3
R["hidden_units"] = (200,) * 3
R["learning_rate"] = 1e-3
R["lr_decay"] = 5e-7

R["varcoe"] = 0.5

R["total_step"] = 5000
R["resamp_epoch"] = 10
R["plot_epoch"] = 500

R["record_path"] = os.path.join("..", "exp3", "E5_BE3_S")

# %% generate data in domain

R["dimension"] = dim = 3
R["area_it"] = 8
R["area_bd"] = 24

# interior data
def rand_it(size):
    x = np.random.rand(size, dim) * 2 - 1
    return x.astype(np.float32)


# boundary data
def rand_bd(size):

    x_bd = np.random.rand(size, dim) * 2 - 1
    ind = np.random.randint(dim, size=size)
    x01 = np.random.randint(2, size=size) * 2 - 1
    for ii in range(size):
        x_bd[ii, ind[ii]] = x01[ii]

    return x_bd.astype(np.float32)


# %% PDE problem
# - (a(x) u'(x))' + c(x) u(x) = f(x)
R["Rad"] = Rad = 0.7
R["Rad0"] = Rad0 = 0.5
R["eps"] = eps1, eps2 = 1, 5
R["kappa"] = kappa1, kappa2 = 0, 0


def u(x):
    r = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    u = np.zeros((x.shape[0], 1))
    for k in range(x.shape[0]):
        if r[k, 0] < Rad0:
            u[k, 0] = r[k, 0] ** 2 / (4 * np.pi * eps1 * Rad0 ** 3) * (
                4 - 3 * r[k, 0] / Rad0
            ) - (1 / eps1 - 1 / eps2) / (4 * np.pi * Rad)
        elif r[k, 0] < Rad:
            u[k, 0] = 1 / (4 * np.pi * eps1 * r[k, 0]) - (1 / eps1 - 1 / eps2) / (
                4 * np.pi * Rad
            )
        else:
            u[k, 0] = 1 / (4 * np.pi * eps2 * r[k, 0])

    return u.astype(np.float32).reshape((-1, 1))


def a(x):
    r2 = np.sum(x * x, axis=1, keepdims=True)
    ind = (r2 < Rad ** 2).astype(np.float32).reshape((-1, 1))
    a = (eps1 - eps2) * ind + eps2

    return a.astype(np.float32).reshape((-1, 1))


def c(x):
    r2 = np.sum(x * x, axis=1, keepdims=True)
    ind = (r2 < Rad ** 2).astype(np.float32).reshape((-1, 1))
    c = (kappa1 ** 2 - kappa2 ** 2) * ind + kappa2 ** 2

    return c.astype(np.float32).reshape((-1, 1))


def f(x):
    r2 = np.sum(x * x, axis=1, keepdims=True)
    ind = (r2 < Rad0 ** 2).astype(np.float32).reshape((-1, 1))
    f = (9 * np.sqrt(r2) - 6 * Rad0) / (np.pi * Rad0 ** 4) * ind

    return f.astype(np.float32).reshape((-1, 1))


# %% save parameters

# prepare folder
if not os.path.isdir(R["record_path"]):
    os.mkdir(R["record_path"])

# save current code
save_file = os.path.join(R["record_path"], "code.py")
shutil.copyfile(__file__, save_file)

# %% set seed
tf.set_random_seed(R["seed"])
tf.random.set_random_seed(R["seed"])
np.random.seed(R["seed"])

# %% get sample to plot
x1, x2, x3 = np.mgrid[-1:1.001:0.02, 0:0.1:1, 0:0.1:1]
x1, x2, x3 = x1.reshape((-1, 1)), x2.reshape((-1, 1)), x3.reshape((-1, 1))
x_samp = np.concatenate([x1, x2, x3], axis=1)

R["x_samp"] = x_samp.astype(np.float32)
R["u_samp_true"] = u(R["x_samp"])

# %% normal neural network
units = (dim,) + R["hidden_units"] + (1,)


def neural_net(x):
    with tf.variable_scope("vscope", reuse=tf.AUTO_REUSE):

        all_y = []
        for k in range(len(R["scale"])):
            scale_y = R["scale"][k] * x

            for i in range(len(units) - 2):
                init_W = np.random.randn(units[i], units[i + 1]).astype(np.float32)
                init_W = init_W * (2 / (units[i] + units[i + 1]) ** R["varcoe"])
                init_b = np.random.randn(units[i + 1]).astype(np.float32)
                init_b = init_b * (2 / (units[i] + units[i + 1]) ** R["varcoe"])

                W = tf.get_variable(name="W" + str(i) + str(k), initializer=init_W)
                b = tf.get_variable(name="b" + str(i) + str(k), initializer=init_b)

                scale_y = act_dict[R["act_name"][i]](tf.matmul(scale_y, W) + b)

            init_W = np.random.randn(units[-2], units[-1]).astype(np.float32)
            init_W = init_W * (2 / (units[i] + units[i + 1]) ** R["varcoe"])
            init_b = np.random.randn(units[-1]).astype(np.float32)
            init_b = init_b * (2 / (units[i] + units[i + 1]) ** R["varcoe"])

            W = tf.get_variable(
                name="W" + str(len(units) - 1) + str(k), initializer=init_W
            )
            b = tf.get_variable(
                name="b" + str(len(units) - 1) + str(k), initializer=init_b
            )

            scale_y = tf.matmul(scale_y, W) + b

            all_y.append(scale_y)

        y = sum(all_y)

    return y


# %% loss and optimizer ("V" for variable)

with tf.variable_scope("vscope", reuse=tf.AUTO_REUSE):

    Vx_it = tf.placeholder(tf.float32, shape=(None, dim))
    Vx_bd = tf.placeholder(tf.float32, shape=(None, dim))

    Vc_it = tf.placeholder(tf.float32, shape=(None, 1))
    Va_it = tf.placeholder(tf.float32, shape=(None, 1))
    Vf_it = tf.placeholder(tf.float32, shape=(None, 1))

    Vu_true_it = tf.placeholder(tf.float32, shape=(None, 1))
    Vu_true_bd = tf.placeholder(tf.float32, shape=(None, 1))

    Vu_it = neural_net(Vx_it)
    Vu_bd = neural_net(Vx_bd)

    Vdu_it = tf.gradients(Vu_it, Vx_it)[0]

    Vloss_it = R["area_it"] * tf.reduce_mean(
        1 / 2 * Va_it * tf.reduce_sum(tf.square(Vdu_it), axis=1, keepdims=True)
        + 1 / 2 * Vc_it * tf.square(Vu_it)
        - Vf_it * Vu_it
    )

    Vloss_bd = R["area_bd"] * tf.reduce_mean(tf.square(Vu_bd - Vu_true_bd))

    Vloss = Vloss_it + R["penalty"] * Vloss_bd

    Verror = R["area_it"] * tf.reduce_mean(tf.square(Vu_it - Vu_true_it))

    learning_rate = tf.placeholder_with_default(input=1e-3, shape=[], name="lr")

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(Vloss)

# %% train model
t0 = time.time()
loss, loss_bd = [], []
error = []
lr = 1 * R["learning_rate"]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(R["total_step"] + 1):

        # generate new data ("N" for number)
        if epoch % R["resamp_epoch"] == 0:

            Nx_it = rand_it(R["size_it"])
            Nx_bd = rand_bd(R["size_bd"])

            Na_it = a(Nx_it)
            Nc_it = c(Nx_it)
            Nf_it = f(Nx_it)

            Nu_bd = u(Nx_bd)
            Nu_it = u(Nx_it)

        # train neural network
        lr = (1 - R["lr_decay"]) * lr

        (_, Nloss, Nloss_bd, Nerror) = sess.run(
            [train_op, Vloss, Vloss_bd, Verror],
            feed_dict={
                Vx_it: Nx_it,
                Vx_bd: Nx_bd,
                Va_it: Na_it,
                Vc_it: Nc_it,
                Vf_it: Nf_it,
                Vu_true_bd: Nu_bd,
                Vu_true_it: Nu_it,
                learning_rate: lr,
            },
        )

        # get current error from reference solution
        loss.append(Nloss)
        loss_bd.append(Nloss_bd)
        error.append(Nerror)

        # show current state
        if epoch % R["plot_epoch"] == 0:

            print("epoch %d, time %.3f" % (epoch, time.time() - t0))
            print("total loss %f, boundary loss %f" % (Nloss, Nloss_bd))
            print("interior error %f" % (Nerror))

            R["time"] = time.time() - t0
            R["loss"] = np.array(loss)
            R["loss_bd"] = np.array(loss_bd)
            R["error"] = np.array(error)

            u_samp = sess.run(Vu_it, feed_dict={Vx_it: R["x_samp"]})
            R["u_samp_" + str(epoch)] = u_samp

            # %% save data
            data_dir = os.path.join(R["record_path"], "data.pkl")
            with open(data_dir, "wb") as file:
                pickle.dump(R, file)

# %% save data
data_dir = os.path.join(R["record_path"], "data.pkl")
with open(data_dir, "wb") as file:
    pickle.dump(R, file)
