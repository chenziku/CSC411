#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:22:05 2018

@author: zikunchen
"""

from pylab import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib import cm
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import cPickle
import os
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter



#Load the MNIST digit data
MNIST = loadmat("mnist_all.mat")

#set seed for reproducing results
SEED = 2018

############## Part 1 ##############
# display sample digis

np.random.seed(SEED)
img_indices = np.random.random_integers(0, 5000, 10)

f, axarr = plt.subplots(10, 10)

# zero
axarr[0,0].imshow(MNIST["train0"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[0,0].axis('off')

axarr[0,1].imshow(MNIST["train0"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[0,1].axis('off')

axarr[0,2].imshow(MNIST["train0"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[0,2].axis('off')

axarr[0,3].imshow(MNIST["train0"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[0,3].axis('off')

axarr[0,4].imshow(MNIST["train0"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[0,4].axis('off')

axarr[0,5].imshow(MNIST["train0"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[0,5].axis('off')

axarr[0,6].imshow(MNIST["train0"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[0,6].axis('off')

axarr[0,7].imshow(MNIST["train0"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[0,7].axis('off')

axarr[0,8].imshow(MNIST["train0"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[0,8].axis('off')

axarr[0,9].imshow(MNIST["train0"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[0,9].axis('off')


# one
axarr[1,0].imshow(MNIST["train1"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[1,0].axis('off')

axarr[1,1].imshow(MNIST["train1"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[1,1].axis('off')

axarr[1,2].imshow(MNIST["train1"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[1,2].axis('off')

axarr[1,3].imshow(MNIST["train1"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[1,3].axis('off')

axarr[1,4].imshow(MNIST["train1"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[1,4].axis('off')

axarr[1,5].imshow(MNIST["train1"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[1,5].axis('off')

axarr[1,6].imshow(MNIST["train1"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[1,6].axis('off')

axarr[1,7].imshow(MNIST["train1"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[1,7].axis('off')

axarr[1,8].imshow(MNIST["train1"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[1,8].axis('off')

axarr[1,9].imshow(MNIST["train1"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[1,9].axis('off')


# two
axarr[2,0].imshow(MNIST["train2"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[2,0].axis('off')

axarr[2,1].imshow(MNIST["train2"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[2,1].axis('off')

axarr[2,2].imshow(MNIST["train2"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[2,2].axis('off')

axarr[2,3].imshow(MNIST["train2"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[2,3].axis('off')

axarr[2,4].imshow(MNIST["train2"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[2,4].axis('off')

axarr[2,5].imshow(MNIST["train2"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[2,5].axis('off')

axarr[2,6].imshow(MNIST["train2"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[2,6].axis('off')

axarr[2,7].imshow(MNIST["train2"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[2,7].axis('off')

axarr[2,8].imshow(MNIST["train2"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[2,8].axis('off')

axarr[2,9].imshow(MNIST["train2"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[2,9].axis('off')

# three
axarr[3,0].imshow(MNIST["train3"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[3,0].axis('off')

axarr[3,1].imshow(MNIST["train3"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[3,1].axis('off')

axarr[3,2].imshow(MNIST["train3"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[3,2].axis('off')

axarr[3,3].imshow(MNIST["train3"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[3,3].axis('off')

axarr[3,4].imshow(MNIST["train3"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[3,4].axis('off')

axarr[3,5].imshow(MNIST["train3"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[3,5].axis('off')

axarr[3,6].imshow(MNIST["train3"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[3,6].axis('off')

axarr[3,7].imshow(MNIST["train3"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[3,7].axis('off')

axarr[3,8].imshow(MNIST["train3"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[3,8].axis('off')

axarr[3,9].imshow(MNIST["train3"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[3,9].axis('off')


# four
axarr[4,0].imshow(MNIST["train4"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[4,0].axis('off')

axarr[4,1].imshow(MNIST["train4"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[4,1].axis('off')

axarr[4,2].imshow(MNIST["train4"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[4,2].axis('off')

axarr[4,3].imshow(MNIST["train4"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[4,3].axis('off')

axarr[4,4].imshow(MNIST["train4"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[4,4].axis('off')

axarr[4,5].imshow(MNIST["train4"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[4,5].axis('off')

axarr[4,6].imshow(MNIST["train4"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[4,6].axis('off')

axarr[4,7].imshow(MNIST["train4"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[4,7].axis('off')

axarr[4,8].imshow(MNIST["train4"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[4,8].axis('off')

axarr[4,9].imshow(MNIST["train4"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[4,9].axis('off')

# five
axarr[5,0].imshow(MNIST["train5"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[5,0].axis('off')

axarr[5,1].imshow(MNIST["train5"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[5,1].axis('off')

axarr[5,2].imshow(MNIST["train5"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[5,2].axis('off')

axarr[5,3].imshow(MNIST["train5"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[5,3].axis('off')

axarr[5,4].imshow(MNIST["train5"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[5,4].axis('off')

axarr[5,5].imshow(MNIST["train5"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[5,5].axis('off')

axarr[5,6].imshow(MNIST["train5"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[5,6].axis('off')

axarr[5,7].imshow(MNIST["train5"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[5,7].axis('off')

axarr[5,8].imshow(MNIST["train5"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[5,8].axis('off')

axarr[5,9].imshow(MNIST["train5"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[5,9].axis('off')

# six
axarr[6,0].imshow(MNIST["train6"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[6,0].axis('off')

axarr[6,1].imshow(MNIST["train6"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[6,1].axis('off')

axarr[6,2].imshow(MNIST["train6"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[6,2].axis('off')

axarr[6,3].imshow(MNIST["train6"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[6,3].axis('off')

axarr[6,4].imshow(MNIST["train6"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[6,4].axis('off')

axarr[6,5].imshow(MNIST["train6"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[6,5].axis('off')

axarr[6,6].imshow(MNIST["train6"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[6,6].axis('off')

axarr[6,7].imshow(MNIST["train6"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[6,7].axis('off')

axarr[6,8].imshow(MNIST["train6"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[6,8].axis('off')

axarr[6,9].imshow(MNIST["train6"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[6,9].axis('off')

# seven
axarr[7,0].imshow(MNIST["train7"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[7,0].axis('off')

axarr[7,1].imshow(MNIST["train7"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[7,1].axis('off')

axarr[7,2].imshow(MNIST["train7"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[7,2].axis('off')

axarr[7,3].imshow(MNIST["train7"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[7,3].axis('off')

axarr[7,4].imshow(MNIST["train7"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[7,4].axis('off')

axarr[7,5].imshow(MNIST["train7"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[7,5].axis('off')

axarr[7,6].imshow(MNIST["train7"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[7,6].axis('off')

axarr[7,7].imshow(MNIST["train7"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[7,7].axis('off')

axarr[7,8].imshow(MNIST["train7"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[7,8].axis('off')

axarr[7,9].imshow(MNIST["train7"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[7,9].axis('off')

# eight
axarr[8,0].imshow(MNIST["train8"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[8,0].axis('off')

axarr[8,1].imshow(MNIST["train8"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[8,1].axis('off')

axarr[8,2].imshow(MNIST["train8"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[8,2].axis('off')

axarr[8,3].imshow(MNIST["train8"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[8,3].axis('off')

axarr[8,4].imshow(MNIST["train8"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[8,4].axis('off')

axarr[8,5].imshow(MNIST["train8"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[8,5].axis('off')

axarr[8,6].imshow(MNIST["train8"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[8,6].axis('off')

axarr[8,7].imshow(MNIST["train8"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[8,7].axis('off')

axarr[8,8].imshow(MNIST["train8"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[8,8].axis('off')

axarr[8,9].imshow(MNIST["train8"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[8,9].axis('off')

# nine
axarr[9,0].imshow(MNIST["train9"][img_indices[0]].reshape((28,28)), cmap=cm.gray)
axarr[9,0].axis('off')

axarr[9,1].imshow(MNIST["train9"][img_indices[1]].reshape((28,28)), cmap=cm.gray)
axarr[9,1].axis('off')

axarr[9,2].imshow(MNIST["train9"][img_indices[2]].reshape((28,28)), cmap=cm.gray)
axarr[9,2].axis('off')

axarr[9,3].imshow(MNIST["train9"][img_indices[3]].reshape((28,28)), cmap=cm.gray)
axarr[9,3].axis('off')

axarr[9,4].imshow(MNIST["train9"][img_indices[4]].reshape((28,28)), cmap=cm.gray)
axarr[9,4].axis('off')

axarr[9,5].imshow(MNIST["train9"][img_indices[5]].reshape((28,28)), cmap=cm.gray)
axarr[9,5].axis('off')

axarr[9,6].imshow(MNIST["train9"][img_indices[6]].reshape((28,28)), cmap=cm.gray)
axarr[9,6].axis('off')

axarr[9,7].imshow(MNIST["train9"][img_indices[7]].reshape((28,28)), cmap=cm.gray)
axarr[9,7].axis('off')

axarr[9,8].imshow(MNIST["train9"][img_indices[8]].reshape((28,28)), cmap=cm.gray)
axarr[9,8].axis('off')

axarr[9,9].imshow(MNIST["train9"][img_indices[9]].reshape((28,28)), cmap=cm.gray)
axarr[9,9].axis('off')

plt.show()


############## Part 2 ##############
# number of digits 
N = 10
# dimension of images
K = 28*28

def output(W, X, b):
    M = X.shape[1]
    B = np.tile(b, M)
    O = np.dot(W.T, X) + B
    return O

def softmax(O):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return np.exp(O)/np.tile(np.sum(exp(O),0), (len(O),1))

def forward(W, X, b):
    O = output(W, X, b)
    P = softmax(O)
    return P

############## Part 3 ##############

# b)
def gradient(X, Y, W, b):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    P = forward(W, X, b)
    dCdO = P - Y
    dCdW = np.dot(X, dCdO.T)
    # change to dCdb
    # dCdB = np.tile(np.sum((y - y_), 1), (M, 1)).T
    dCdb = np.sum(dCdO, 1).reshape(N, 1)
    return dCdW, dCdb

def NLL(P, Y):
    return -np.sum(Y*np.log(P))

def check_gradient(X, Y, W, b, i, j, h=1e-7):
    deltaij = np.zeros(W.shape)
    deltaij[i, j] = h
    
    P = forward(W, X, b)
    cost = NLL(P, Y)
    
    P_h = forward(W+deltaij, X, b)
    cost_h = NLL(P_h, Y)
    
    dCdWij = gradient(X, Y, W, b)[0][i, j]
    finite_diff = (cost_h - cost)/h
    
    #dj = dJ(X, Y, W)[i, j]
    
    print 'W[%1d, %1d]'%(i, j)
    print '--------------------------------------'
    print 'dCdwij: %.7f\nfinite-difference approximation: %.7f' % (dCdWij, finite_diff)
    print 'absolute difference: %.7f\n' % (abs(dCdWij - finite_diff))


# load sample weights and biases
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"]

# initialize weights and biases         
np.random.seed(SEED)
np.random.shuffle(W0)
init_W = W0[:, :N]

np.random.seed(SEED)
np.random.shuffle(b0)
init_b = b0[:N].reshape((N, 1))



# split training and validation sets with last element being the label in each digit
X_train = np.vstack(( \
            MNIST["train0"][:int(len(MNIST["train0"])*0.8)], \
            MNIST["train1"][:int(len(MNIST["train1"])*0.8)], \
            MNIST["train2"][:int(len(MNIST["train2"])*0.8)], \
            MNIST["train3"][:int(len(MNIST["train3"])*0.8)], \
            MNIST["train4"][:int(len(MNIST["train4"])*0.8)], \
            MNIST["train5"][:int(len(MNIST["train5"])*0.8)], \
            MNIST["train6"][:int(len(MNIST["train6"])*0.8)], \
            MNIST["train7"][:int(len(MNIST["train7"])*0.8)], \
            MNIST["train8"][:int(len(MNIST["train8"])*0.8)], \
            MNIST["train9"][:int(len(MNIST["train9"])*0.8)], \
          )).T / 255.
                     
X_valid = np.vstack(( \
            MNIST["train0"][int(len(MNIST["train0"])*0.8):], \
            MNIST["train1"][int(len(MNIST["train1"])*0.8):], \
            MNIST["train2"][int(len(MNIST["train2"])*0.8):], \
            MNIST["train3"][int(len(MNIST["train3"])*0.8):], \
            MNIST["train4"][int(len(MNIST["train4"])*0.8):], \
            MNIST["train5"][int(len(MNIST["train5"])*0.8):], \
            MNIST["train6"][int(len(MNIST["train6"])*0.8):], \
            MNIST["train7"][int(len(MNIST["train7"])*0.8):], \
            MNIST["train8"][int(len(MNIST["train8"])*0.8):], \
            MNIST["train9"][int(len(MNIST["train9"])*0.8):], \
          )).T / 255.

# construct labels for training and validation sets                     
Y_train = np.zeros([N, X_train.shape[1]])
Y_valid = np.zeros([N, X_valid.shape[1]])
curr_train = 0
curr_valid = 0
for set_name in np.sort(MNIST.keys()):
    if set_name[:5] == 'train':
        length = len(MNIST[set_name])
        train_len = int(length*0.8)
        valid_len = length - train_len
        
        Y_train[int(set_name[-1]), curr_train:curr_train + train_len] = 1
        Y_valid[int(set_name[-1]), curr_valid:curr_valid + valid_len] = 1
        
        curr_train += train_len
        curr_valid += valid_len
        
train_shuffle = np.vstack((X_train, Y_train))
valid_shuffle = np.vstack((X_valid, Y_valid))

train_shuffle = train_shuffle.T
valid_shuffle = valid_shuffle.T

np.random.seed(SEED)
np.random.shuffle(train_shuffle)

np.random.seed(SEED)
np.random.shuffle(valid_shuffle)

X_train = train_shuffle.T[:K, :]
Y_train = train_shuffle.T[K:, :]
X_valid = valid_shuffle.T[:K, :]
Y_valid = valid_shuffle.T[K:, :]


# check gradients
check_gradient(X_train, Y_train, init_W, init_b, 456, 3, h=1e-7)
check_gradient(X_train, Y_train, init_W, init_b, 47, 0, h=1e-7)
check_gradient(X_train, Y_train, init_W, init_b, 209, 5, h=1e-7)
check_gradient(X_train, Y_train, init_W, init_b, 653, 9, h=1e-7)
check_gradient(X_train, Y_train, init_W, init_b, 200, 4, h=1e-7)

  

############## Part 4 ##############

def vanilla_train(X_train, X_valid, Y_train, Y_valid, init_W, init_b, alpha = 1e-6, \
                  count=5, EPS=1e-5, freq=100, max_iter=300000): 
    global W_vanilla
    global b_vanilla
    
    # gradient descent
    prev_W = init_W-10*EPS
    W = init_W.copy()
    
    
    prev_b = init_b-10*EPS
    b = init_b.copy()
    
    iter  = 0

    while np.linalg.norm(W - prev_W) > EPS and iter < max_iter \
        and np.count_nonzero(np.diff(valid_costs)[-5:]>0) < count:
            
        prev_W = W_vanilla.copy()
        prev_b = b_vanilla.copy()
        
        P = forward(W, X_train, b)
        
        dCdW, dCdb = gradient(X_train, Y_train, W, b)
        
        W -= alpha*dCdW
        b -= alpha*dCdb
        
        W_vanilla = W
        b_vanilla = b
        
        if iter % freq == 0:
            
            P_train = forward(W_vanilla, X_train, b)
            t_cost = NLL(P_train, Y_train)
            dCdW, dCdb = gradient(X_train, Y_train, W, b)
            
            print "Epoch ", iter
            
            # print gradient
            print "dCdw =  ", dCdW, "\n"
            print "dCdb =  ", dCdb, "\n" 

            # report costs
            P_valid = forward(W, X_valid, b)
            v_cost = NLL(P_valid, Y_valid)

            train_costs.append(t_cost)
            valid_costs.append(v_cost)

            print '\ntraining cost: %.7f\nvalidation cost: %.7f\n'% (t_cost, v_cost)

            epoch.append(iter)
            
        iter += 1


# vanilla training 
train_costs = list()
valid_costs = list()
epoch = list()
W_vanilla = init_W.copy()
b_vanilla = init_b.copy()
vanilla_train(X_train, X_valid, Y_train, Y_valid, init_W, init_b, alpha = 1e-5, EPS=1e-5, freq=100, max_iter=500000)


# Plot Learing Curve
plt.plot(epoch, train_costs, color='blue')
plt.plot(epoch, valid_costs, color='green')
plt.title('Learning Curve')
plt.ylabel('cost')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# Display Weights

f, axarr = plt.subplots(2, 5)

# first row zero to four
axarr[0,0].imshow(W_vanilla[:,1].reshape((28,28)), cmap=cm.coolwarm)
axarr[0,0].axis('off')

axarr[0,1].imshow(W_vanilla[:,1].reshape((28,28)), cmap=cm.coolwarm)
axarr[0,1].axis('off')

axarr[0,2].imshow(W_vanilla[:,2].reshape((28,28)), cmap=cm.coolwarm)
axarr[0,2].axis('off')

axarr[0,3].imshow(W_vanilla[:,3].reshape((28,28)), cmap=cm.coolwarm)
axarr[0,3].axis('off')

axarr[0,4].imshow(W_vanilla[:,4].reshape((28,28)), cmap=cm.coolwarm)
axarr[0,4].axis('off')

# second row five to nine
axarr[1,0].imshow(W_vanilla[:,5].reshape((28,28)), cmap=cm.coolwarm)
axarr[1,0].axis('off')

axarr[1,1].imshow(W_vanilla[:,6].reshape((28,28)), cmap=cm.coolwarm)
axarr[1,1].axis('off')

axarr[1,2].imshow(W_vanilla[:,7].reshape((28,28)), cmap=cm.coolwarm)
axarr[1,2].axis('off')

axarr[1,3].imshow(W_vanilla[:,8].reshape((28,28)), cmap=cm.coolwarm)
axarr[1,3].axis('off')

axarr[1,4].imshow(W_vanilla[:,9].reshape((28,28)), cmap=cm.coolwarm)
axarr[1,4].axis('off')

plt.show()

############## Part 5 ##############

def momentum_train(X_train, X_valid, Y_train, Y_valid, init_W, init_b, alpha = 1e-6, \
                   gamma = 0.99, EPS=1e-5, count = 5, freq=100, max_iter=10000):
    
    prev_W = init_W-10*EPS
    W = init_W.copy()
    
    prev_b = init_b-10*EPS
    b = init_b.copy()
    
    global W_mom
    global b_mom
    
    
    change_W = np.zeros(W.shape)
    change_b = np.zeros(b.shape)
    
    iter  = 0
    
    
    # gradient descent with momentum
    while np.linalg.norm(W - prev_W) > EPS and iter < max_iter \
        and np.count_nonzero(np.diff(valid_costs_mom)[-5:]>0) < count:
        prev_W = W.copy()
        prev_b = b.copy()
        
        prev_change_W = change_W
        prev_change_b = change_b
        
        P = forward(W, X_train, b)
        
        dCdW, dCdb = gradient(X_train, Y_train, W, b)
        
        change_W = gamma * prev_change_W + alpha * dCdW
        change_b = gamma * prev_change_b + alpha * dCdb
        
        W -= change_W
        b -= change_b
        
        W_mom = W
        b_mom = b
        
        
        if iter % freq == 0:
            P_train = forward(W, X_train, b)
            t_cost = NLL(P_train, Y_train)
            dCdW, dCdb = gradient(X_train, Y_train, W, b)
            
            print "Epoch ", iter
            
            # print gradient
            print "dCdw =  ", dCdW, "\n"
            print "dCdb =  ", dCdb, "\n" 

            # report costs
            P_valid = forward(W, X_valid, b)
            v_cost = NLL(P_valid, Y_valid)

            train_costs_mom.append(t_cost)
            valid_costs_mom.append(v_cost)

            print '\ntraining cost: %.7f\nvalidation cost: %.7f\n'% (t_cost, v_cost)
            
            epoch_mom.append(iter)
        iter += 1

train_costs_mom = list()
valid_costs_mom = list()
epoch_mom = list()
momentum_train(X_train, X_valid, Y_train, Y_valid, init_W, init_b, alpha = 1e-5, gamma = 0.99, EPS=1e-6, freq=100, max_iter=500000)

# Plot Learing Curve
plt.plot(epoch_mom, train_costs_mom, color='blue')
plt.plot(epoch_mom, valid_costs_mom, color='green')
plt.title('Learning Curve')
plt.ylabel('cost')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


def accuracy(W, b, X, Y):
    total = X.shape[1]
    P = forward(W, X, b)
    count = 0
    for i in range(total):
        if np.argmax(P[:, i]) == np.argmax(Y[:, i]):
            count += 1
    return count/float(total)

############## Part 6 ##############
# 6 a)

# reconstruct smaller set for contour plot
X_contour = np.vstack(( \
            MNIST["train0"][:400], \
            MNIST["train1"][:400], \
            MNIST["train2"][:400], \
            MNIST["train3"][:400], \
            MNIST["train4"][:400], \
            MNIST["train5"][:400], \
            MNIST["train6"][:400], \
            MNIST["train7"][:400], \
            MNIST["train8"][:400], \
            MNIST["train9"][:400], \
          )).T / 255.

# construct labels for training and validation sets                     
Y_contour = np.zeros([N, X_contour.shape[1]])
curr = 0
for set_name in np.sort(MNIST.keys()):
    if set_name[:5] == 'train':
        Y_contour[int(set_name[-1]), curr:(curr + 400)] = 1
        curr += 400
        
contour_shuffle = np.vstack((X_contour, Y_contour))
contour_shuffle = contour_shuffle.T

np.random.seed(SEED)
np.random.shuffle(contour_shuffle)

X_contour = contour_shuffle.T[:K, :]
Y_contour = contour_shuffle.T[K:, :]


# plot contour
s = 5
step = 1


i1 = 350
j1 = 2
i2 = 350
j2 = 5

W_contour = W_mom.copy()
b_contour = b_mom.copy()

init_w1 = W_contour[i1, j1] - step*(s/2)
init_w2 = W_contour[i2, j2] - step*(s/2)

W1 = np.zeros(s)
W2 = np.zeros(s)
C = np.zeros((s,s))

for i in range(5):
    for j in range(5):
        
        w1 = init_w1 + i * step
        w2 = init_w2 + j * step
        
        W1[i] = w1
        W2[j] = w2
        
        W_contour[i1, j1] = w1
        W_contour[i2, j2] = w2
        
        C[i,j] = NLL(forward(W_contour, X_contour, b_contour), Y_contour)
    
W11, W22 = meshgrid(W1, W2)
CS = contour(W11, W22, C, camp=cm.coolwarm)
clabel(CS, inline=1, fontsize=10)
title('Contour plot')
plt.xlabel('W1 ('+ str(i1) + ',' + str(j1) + ')')
plt.ylabel('W2 ('+ str(i2) + ',' + str(j2) + ')')
show()

        
   
# 6 b)  
def vanilla_traj(X_train, Y_train, init_W, init_b, alpha = 1e-5, max_iter=20): 
    global gd_traj
    
    # gradient descent
    W = init_W.copy()
    
    b = init_b.copy()
    
    iter  = 0
    while iter < max_iter:
        
        P = forward(W, X_train, b)
        
        dCdW, dCdb = gradient(X_train, Y_train, W, b)
        
        W[i1, j1] -= alpha*dCdW[i1, j1]
        W[i2, j2] -= alpha*dCdW[i2, j2]
        
        gd_traj.append((W[i1, j1], W[i2, j2]))
        
        P_train = forward(W, X_train, b)
        t_cost = NLL(P_train, Y_train)
        print "Epoch ", iter

        # report costs
        print 'training cost: %.7f\n'% (t_cost)

        iter += 1        
        
        
i1 = 350
j1 = 2
i2 = 350
j2 = 5

init_w1 = init_W[i1, j1]
init_w2 = init_W[i2, j2]

init_w1 = -5
init_w2 = -2.5


gd_traj = [(init_w1, init_w2)]

init_W_contour = W_mom.copy()
init_b_contour = b_mom.copy()

init_W_contour[i1, j1] = init_w1
init_W_contour[i2, j2] = init_w2

vanilla_traj(X_contour, Y_contour, init_W_contour, init_b_contour, \
             alpha = 2e-2, max_iter=20)

w1s = np.arange(-6, 6, 1)
w2s = np.arange(-6, 6, 1)
w1z, w2z = np.meshgrid(w1s, w2s)
C = np.zeros([w1s.size, w2s.size])
for i, w1 in enumerate(w1s):
    for j, w2 in enumerate(w2s):
        W_contour[i1, j1] = w1
        W_contour[i2, j2] = w2
        C[i,j] = NLL(forward(W_contour, X_contour, b_contour), Y_contour)
   
CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
plt.legend(loc='top left')
plt.title('Contour plot')
plt.xlabel('W1 ('+ str(i1) + ',' + str(j1) + ')')
plt.ylabel('W2 ('+ str(i2) + ',' + str(j2) + ')')
plt.show()     
        
        
# 6 c)
def momentum_traj(X_train, Y_train, init_W, init_b, alpha = 1e-5, gamma = 0.99, max_iter=20):
    global mo_traj
    
    W = init_W.copy()
    b = init_b.copy()
    
    change_W = np.zeros(W.shape)
    
    iter  = 0
    while iter < max_iter: 
        
        prev_change_W = change_W
        
        P = forward(W, X_train, b)
        
        dCdW, dCdb = gradient(X_train, Y_train, W, b)
        
        change_W = gamma * prev_change_W + alpha * dCdW

        
        W[i1, j1] -= change_W[i1, j1] 
        W[i2, j2] -= change_W[i2, j2] 
            
        mo_traj.append((W[i1, j1], W[i2, j2]))
        
        P_train = forward(W, X_train, b)
        t_cost = NLL(P_train, Y_train)
        print "Epoch ", iter

        # report costs
        print 'training cost: %.7f\n'% (t_cost)

        iter += 1

i1 = 350
j1 = 2
i2 = 350
j2 = 5        


init_w1 = init_W[i1, j1]
init_w2 = init_W[i2, j2]

init_w1 = -5
init_w2 = -2.5

mo_traj = [(init_w1, init_w2)]

init_W_contour = W_mom.copy()
init_b_contour = b_mom.copy()

init_W_contour[i1, j1] = init_w1
init_W_contour[i2, j2] = init_w2

momentum_traj(X_contour, Y_contour, init_W_contour, init_b_contour, \
             alpha = 1e-3, gamma = 0.95, max_iter=20)
   
w1s = np.arange(-6, 6, 1)
w2s = np.arange(-6, 6, 1)
w1z, w2z = np.meshgrid(w1s, w2s)
C = np.zeros([w1s.size, w2s.size])
for i, w1 in enumerate(w1s):
    for j, w2 in enumerate(w2s):
        W_contour[i1, j1] = w1
        W_contour[i2, j2] = w2
        C[i,j] = NLL(forward(W_contour, X_contour, b_contour), Y_contour)
   
CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
plt.legend(loc='top left')
plt.title('Contour plot')
plt.xlabel('W1 ('+ str(i1) + ',' + str(j1) + ')')
plt.ylabel('W2 ('+ str(i2) + ',' + str(j2) + ')')
plt.show()

    
    
    
# 6 e)
# show scenerio where momentum is better
i1 = 233
j1 = 1
i2 = 301
j2 = 4  
        
init_w1 = init_W[i1, j1]
init_w2 = init_W[i2, j2]

init_w1 = -10
init_w2 = -2.5


gd_traj = [(init_w1, init_w2)]
mo_traj = [(init_w1, init_w2)]

init_W_contour = W_mom.copy()
init_b_contour = b_mom.copy()

init_W_contour[i1, j1] = init_w1
init_W_contour[i2, j2] = init_w2

vanilla_traj(X_contour, Y_contour, init_W_contour, init_b_contour, \
             alpha = 3e-1, max_iter=20)

momentum_traj(X_contour, Y_contour, init_W_contour, init_b_contour, \
             alpha = 4e-2, gamma = 0.95, max_iter=20)
   
w1s = np.arange(-10, 10, 1)
w2s = np.arange(-10, 10, 1)
w1z, w2z = np.meshgrid(w1s, w2s)
C = np.zeros([w1s.size, w2s.size])
for i, w1 in enumerate(w1s):
    for j, w2 in enumerate(w2s):
        W_contour[i1, j1] = w1
        W_contour[i2, j2] = w2
        C[i,j] = NLL(forward(W_contour, X_contour, b_contour), Y_contour)
   
CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
plt.legend(loc='top left')
plt.title('Contour plot')
plt.xlabel('W1 ('+ str(i1) + ',' + str(j1) + ')')
plt.ylabel('W2 ('+ str(i2) + ',' + str(j2) + ')')
plt.show()

# show scenerio where no momentum is better
i1 = 400
j1 = 3
i2 = 350
j2 = 6

init_w1 = init_W[i1, j1]
init_w2 = init_W[i2, j2]

init_w1 = -10
init_w2 = -2.5


gd_traj = [(init_w1, init_w2)]
mo_traj = [(init_w1, init_w2)]

init_W_contour = W_mom.copy()
init_b_contour = b_mom.copy()

init_W_contour[i1, j1] = init_w1
init_W_contour[i2, j2] = init_w2

vanilla_traj(X_contour, Y_contour, init_W_contour, init_b_contour, \
             alpha = 5e-2, max_iter=20)

momentum_traj(X_contour, Y_contour, init_W_contour, init_b_contour, \
             alpha = 1e-2, gamma = 0.9, max_iter=20)

w1s = np.arange(-10, 10, 1)
w2s = np.arange(-10, 10, 1)
w1z, w2z = np.meshgrid(w1s, w2s)
C = np.zeros([w1s.size, w2s.size])
for i, w1 in enumerate(w1s):
    for j, w2 in enumerate(w2s):
        W_contour[i1, j1] = w1
        W_contour[i2, j2] = w2
        C[i,j] = NLL(forward(W_contour, X_contour, b_contour), Y_contour)
   
CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
plt.legend(loc='top left')
plt.title('Contour plot')
plt.xlabel('W1 ('+ str(i1) + ',' + str(j1) + ')')
plt.ylabel('W2 ('+ str(i2) + ',' + str(j2) + ')')
plt.show()

