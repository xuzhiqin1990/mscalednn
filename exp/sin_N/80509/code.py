# !python3
# coding: utf-8
# author: Ziqi Liu and Zhi-Qin John Xu
# Reference: Ziqi Liu，Wei Cai，Zhi-Qin John Xu. Multi-scale Deep Neural Network (MscaleDNN)
# for Solving Poisson-Boltzmann Equation in Complex Domains[J]. 2020. arXiv:2007.11207 (CiCP)

import os
import time
import shutil
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from my_act import act_dict

import matplotlib
matplotlib.use('Agg')      
Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88 - Leftp
Heightp = 0.9 - Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]
 
def save_fig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=0,isShowPic=0):# Save the figure
    if isax==1:
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        ax.set_position(pos, which='both')
    fnm = '%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm = '%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic==1:
        pltm.show() 
    elif isShowPic==-1:
        return
    else:
        pltm.close()
        

# %% parameters
R = {}

R["seed"] = 0

R["size_it"] = 1000
R["size_bd"] = 100

R["penalty"] = 1e3

R["act_name"] = ("sin",) * 3
R["hidden_units"] = (180,) * 3
R["learning_rate"] = 1e-4
R["lr_decay"] = 5e-7

R["varcoe"] = 0.5

R["total_step"] = 5000
R["resamp_epoch"] = 1
R["plot_epoch"] = 500

R["record_path"] = os.path.join("..", "exp", "sin_N",str(np.random.randint(10000,99999)))

# %% generate data in domain

R["dimension"] = dim = 1
R["area_it"] = 2
R["area_bd"] = 1

# interior data
def rand_it(size):
    x_it = np.random.rand(size, dim) * 2 - 1
    return x_it.astype(np.float32)


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
R["mu"] = mu = 6 * np.pi


def u(x):
    u = np.sin(mu * x)
    return u.astype(np.float32).reshape((-1, 1))


def a(x):
    a = np.ones((x.shape[0], 1))
    return a.astype(np.float32).reshape((-1, 1))


def c(x):
    c = np.zeros((x.shape[0], 1))
    return c.astype(np.float32).reshape((-1, 1))


def f(x):
    f = mu * mu * np.sin(mu * x)
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

# %% get test points
x_test = np.linspace(-1, 1, 200 +1).reshape((-1, 1))

R["x_test"] = x_test.astype(np.float32)
R["u_test_true"] = u(R["x_test"])

# %% get sample points
x_samp = np.linspace(-1, 1, 200 +1).reshape((-1, 1))

R["x_samp"] = x_samp.astype(np.float32)
R["u_samp_true"] = u(R["x_samp"])

# %% normal neural network
units = (dim,) + R["hidden_units"] + (1,)


def neural_net(x):
    with tf.variable_scope("vscope", reuse=tf.AUTO_REUSE):

        y = x

        for i in range(len(units) - 2):
            init_W = np.random.randn(units[i], units[i + 1]).astype(np.float32)
            init_W = init_W * (2 / (units[i] + units[i + 1]) ** R["varcoe"])
            init_b = np.random.randn(units[i + 1]).astype(np.float32)
            init_b = init_b * (2 / (units[i] + units[i + 1]) ** R["varcoe"])

            W = tf.get_variable(name="W" + str(i), initializer=init_W)
            b = tf.get_variable(name="b" + str(i), initializer=init_b)

            y = act_dict[R["act_name"][i]](tf.matmul(y, W) + b)

        init_W = np.random.randn(units[-2], units[-1]).astype(np.float32)
        init_W = init_W * (2 / (units[i] + units[i + 1]) ** R["varcoe"])
        init_b = np.random.randn(units[-1]).astype(np.float32)
        init_b = init_b * (2 / (units[i] + units[i + 1]) ** R["varcoe"])

        W = tf.get_variable(name="W" + str(len(units) - 1), initializer=init_W)
        b = tf.get_variable(name="b" + str(len(units) - 1), initializer=init_b)

        y = tf.matmul(y, W) + b

    return y


# %% loss and optimizer ("V" for variable)
with tf.variable_scope("vscope", reuse=tf.AUTO_REUSE):

    Vx_it = tf.placeholder(tf.float32, shape=(None, dim))
    Vx_bd = tf.placeholder(tf.float32, shape=(None, dim))

    Va_it = tf.placeholder(tf.float32, shape=(None, 1))
    Vc_it = tf.placeholder(tf.float32, shape=(None, 1))
    Vf_it = tf.placeholder(tf.float32, shape=(None, 1))

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

        # train neural network
        lr = (1 - R["lr_decay"]) * lr

        _, Nloss, Nloss_bd = sess.run(
            [train_op, Vloss, Vloss_bd],
            feed_dict={
                Vx_it: Nx_it,
                Vx_bd: Nx_bd,
                Va_it: Na_it,
                Vc_it: Nc_it,
                Vf_it: Nf_it,
                Vu_true_bd: Nu_bd,
                learning_rate: lr,
            },
        )

        # get test error
        R["u_test"] = sess.run(Vu_it, feed_dict={Vx_it: R["x_test"]})
        Nerror = R["area_it"] * np.mean((R["u_test"] - R["u_test_true"]) ** 2)

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
            
            plt.figure()
            ax = plt.gca()
            plt.semilogy(R["error"])
            plt.title('error',fontsize=15)         
            fntmp = os.path.join(R["record_path"], 'error') 
            save_fig(plt,fntmp,ax=ax,isax=1,iseps=0)
            
            plt.figure()
            ax = plt.gca()
            plt.plot(R["loss"])
            plt.title('loss',fontsize=15)         
            fntmp = os.path.join(R["record_path"], 'loss') 
            save_fig(plt,fntmp,ax=ax,isax=1,iseps=0)
            
            plt.figure()
            ax = plt.gca()
            plt.semilogy(R["loss_bd"])
            plt.title('loss_bd',fontsize=15)         
            fntmp = os.path.join(R["record_path"], 'loss_bd') 
            save_fig(plt,fntmp,ax=ax,isax=1,iseps=0)
            
            u_samp = sess.run(Vu_it, feed_dict={Vx_it: R["x_samp"]})
            R["u_samp_" + str(epoch)] = u_samp
            
            if R['dimension']==1: 
                plt.figure()
                ax = plt.gca()
                plt.plot(R["x_samp"], R["u_samp_"+str(epoch)], 'r--',label='dnn')
                plt.plot(R["x_samp"], R["u_samp_true"], 'b-',label='true')
                plt.title('u',fontsize=15)        
                plt.legend(fontsize=18) 
                fntmp = os.path.join(R["record_path"], 'u_epoch_%s'%(epoch)) 
                save_fig(plt,fntmp,ax=ax,isax=1,iseps=0)
            

            # %% save data
            data_dir = os.path.join(R["record_path"], "data.pkl")
            with open(data_dir, "wb") as file:
                pickle.dump(R, file)

# %% save data
data_dir = os.path.join(R["record_path"], "data.pkl")
with open(data_dir, "wb") as file:
    pickle.dump(R, file)
