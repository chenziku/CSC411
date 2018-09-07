#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 20:25:23 2018

@author: zikunchen
"""

import re
import numpy as np
from matplotlib import cm
import scipy
from scipy import arange
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from scipy.misc import imshow
import os
import urllib
import hashlib
import matplotlib.image as mpimg
from torch.autograd import Variable
import torch
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import mglearn
from sklearn import tree
import pydotplus
from scipy import misc
import io
import StringIO
import math


########### Construction of Sets ########### 
SEED = 2018

fake = np.array(0)
for line in open("clean_fake.txt").readlines():
    fake = np.append(fake, line.split('\n')[0])
fake = fake[1:]
    
real = np.array(0)
for line in open("clean_real.txt").readlines():
    real = np.append(real, line.split('\n')[0])
real = real[1:]

    
np.random.seed(SEED)
fake_index = np.arange(len(fake))
np.random.shuffle(fake_index)

np.random.seed(SEED)
real_index = np.arange(len(real))
np.random.shuffle(real_index)


fake_split1 = int(round(0.7*len(fake)))
fake_split2 = fake_split1 + int(round(0.15*len(fake)))

fake_train_i = fake_index[:fake_split1]
fake_valid_i = fake_index[fake_split1:fake_split2]
fake_test_i = fake_index[fake_split2:]


real_split1 = int(round(0.7*len(real)))
real_split2 = real_split1 + int(round(0.15*len(real)))

real_train_i = real_index[:real_split1]
real_valid_i = real_index[real_split1:real_split2]
real_test_i = real_index[real_split2:]


# construct dictionary of frequencies of words in the training set of fake titles      
fake_unique_words = []
fake_train = fake[fake_train_i]            
for title in fake_train:
    for word in title.split(' '):
        if word not in fake_unique_words:
            fake_unique_words.append(word)
            
fake_train_freq = {}
for word in fake_unique_words:
    for title in fake_train:
        if word in title.split(' ') and word in fake_train_freq.keys():
            fake_train_freq[word] += 1
            continue
        elif word in title.split(' ') and word not in fake_train_freq.keys():
            fake_train_freq[word] = 1
            continue
    
     
# construct dictionary of frequencies of words in the training set of real titles
real_unique_words = []
real_train = real[real_train_i]            
for title in real_train:
    for word in title.split(' '):
        if word not in real_unique_words:
            real_unique_words.append(word)
            
real_train_freq = {}
for word in real_unique_words:
    for title in real_train:
        if word in title.split(' ') and word in real_train_freq.keys():
            real_train_freq[word] += 1
            continue
        elif word in title.split(' ') and word not in real_train_freq.keys():
            real_train_freq[word] = 1
            continue
    

  
########### Part 1 ########### 

# display most frequent words in real and fake headlines in training set
            
for key, value in sorted(fake_train_freq.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)

for key, value in sorted(real_train_freq.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)


########### Part 2 Naive Bayes ###########
    
# list of unique words in the training set
train_unique_words = []
train = np.hstack((fake_train, real_train))        
for title in train:
    for word in title.split(' '):
        if word not in train_unique_words:
            train_unique_words.append(word)

fake_train_size = fake_train_i.size
real_train_size = real_train_i.size

p_fake_train = fake_train_size / float(fake_train_size + real_train_size)
p_real_train = 1 - p_fake_train

def perdict(title, m, p_hat):
    sum_r = 0
    sum_f = 0
  
    in_title = []
    title_list = title.split(' ')
    for word in train_unique_words:
        if word in title_list:
            if word in fake_train_freq.keys():
                p_xi_f = (fake_train_freq[word] + m * p_hat) / float(fake_train_size + m)
                sum_f += np.log(p_xi_f)
            else:
                p_xi_f = (m * p_hat) / float(fake_train_size + m)
                sum_f += np.log(p_xi_f)

            if word in real_train_freq.keys():
                p_xi_r = (real_train_freq[word] + m * p_hat) / float(real_train_size + m)
                sum_r += np.log(p_xi_r)
            else:
                p_xi_r = (m * p_hat) / float(real_train_size + m)
                sum_r += np.log(p_xi_r)

    sum_f += np.log(p_fake_train)
    sum_r += np.log(p_real_train)
    return (sum_r - sum_f) >= 0


def NB_accuracy(index_fake, index_real, m, p_hat):
    count = 0
    for title in fake[index_fake]:
        if perdict(title, m, p_hat) == False:
            count += 1 
    for title in real[index_real]:
        if perdict(title, m, p_hat) == True:
            count += 1 
    return count / float(index_fake.size + index_real.size)

# grid search for the best m and p_hat    
p_hat = 0
best_acc = 0
for m in range(1, 5):
    for i in range(1, 20):
        p_hat = 0.001 * i
        acc = NB_accuracy(fake_valid_i, real_valid_i, m, p_hat)
        if acc > best_acc:
            best_acc = acc
            best_p = p_hat
            best_m = m
            print best_acc
            print best_p
            print best_m

# final performance
print '\n--------- Final Performance ---------'
print '\nbest m: %d\nbest p: %.3f' % (best_m, best_p)
print '\ntraining accuracy: %.2f%%\nvalidation accuracy: %.2f%%\ntest accuracy: %.2f%%\n' \
    % (NB_accuracy(fake_train_i, real_train_i, best_m, best_p)*100, \
       NB_accuracy(fake_valid_i, real_valid_i, best_m, best_p)*100, \
       NB_accuracy(fake_test_i, real_test_i, best_m, best_p)*100)



########### Part 3 ###########
# 3a) 
prw = {}
prn = {}
pfw = {}
pfn = {}

for word in train_unique_words:
    
    if word in fake_train_freq.keys():
        p_w_fake = fake_train_freq[word]/float(fake_train_size)
    else:
        p_w_fake = best_m * best_p / float(fake_train_size + best_m)
        
    if word in real_train_freq.keys():
        p_w_real = real_train_freq[word]/float(real_train_size)
    else:
        p_w_real = best_m * best_p / float(real_train_size + best_m)
        
    p_real_w = (p_w_real * p_real_train) \
                / (p_w_real * p_real_train + p_w_fake * p_fake_train)
    p_fake_w = 1 - p_real_w
    p_real_nonw = ((1 - p_w_real) * p_real_train) \
                / ((1 - p_w_real) * p_real_train + (1 - p_w_fake) * p_fake_train)
    p_fake_nonw = 1 - p_real_nonw
    
    prw[word] = p_real_w
    prn[word] = p_real_nonw
    pfw[word] = p_fake_w
    pfn[word] = p_fake_nonw

 
for key, value in sorted(prw.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)
    
for key, value in sorted(prn.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)
    
for key, value in sorted(pfw.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)
    
for key, value in sorted(pfn.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)


# 3b) nonstop words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

prw_nonstop = {}
pfw_nonstop = {}

for word in train_unique_words:
    if word not in ENGLISH_STOP_WORDS:
        if word in fake_train_freq.keys():
            p_w_fake = fake_train_freq[word]/float(fake_train_size)
        else:
            p_w_fake = best_m * best_p / float(fake_train_size + best_m)
            
        if word in real_train_freq.keys():
            p_w_real = real_train_freq[word]/float(real_train_size)
        else:
            p_w_real = best_m * best_p / float(real_train_size + best_m)
            
        p_real_w = (p_w_real * p_real_train) \
                    / (p_w_real * p_real_train + p_w_fake * p_fake_train)
        p_fake_w = 1 - p_real_w
    
        prw_nonstop[word] = p_real_w
        pfw_nonstop[word] = p_fake_w

for key, value in sorted(prw_nonstop.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)
    
for key, value in sorted(pfw_nonstop.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)



########### Part 4 - Logistic Regression ###########

# encode training set into vector of 0s and 1s
dim_x = len(train_unique_words)
dim_out = 1

train_unique_words = np.array(train_unique_words)

train_x = np.zeros(dim_x)
for title in fake[fake_train_i]:
    x = np.zeros(dim_x)
    for word in title.split(' '):
        index = np.where(train_unique_words == word)[0]
        x[index] = 1
    train_x = np.vstack((train_x, x))

for title in real[real_train_i]:
    x = np.zeros(dim_x)
    for word in title.split(' '):
        index = np.where(train_unique_words == word)[0]
        x[index] = 1
    train_x = np.vstack((train_x, x))  
train_x = train_x[1:]

# construct labels
train_y =np.vstack((np.tile(np.array(0), (fake_train_i.size, 1)), \
                   np.tile(np.array(1), (real_train_i.size, 1))))

# encode validation set into vector of 0s and 1s
valid_x = np.zeros(dim_x)
for title in fake[fake_valid_i]:
    x = np.zeros(dim_x)
    for word in title.split(' '):
        if word in train_unique_words:
            index = np.where(train_unique_words == word)[0]
            x[index] = 1
    valid_x = np.vstack((valid_x, x))

for title in real[real_valid_i]:
    x = np.zeros(dim_x)
    for word in title.split(' '):
        if word in train_unique_words:
            index = np.where(train_unique_words == word)[0]
            x[index] = 1
    valid_x = np.vstack((valid_x, x))  
valid_x = valid_x[1:]

# construct labels
valid_y =np.vstack((np.tile(np.array(0), (fake_valid_i.size, 1)), \
                   np.tile(np.array(1), (real_valid_i.size, 1))))

# encode test set into vector of 0s and 1s
test_x = np.zeros(dim_x)
for title in fake[fake_test_i]:
    x = np.zeros(dim_x)
    for word in title.split(' '):
        if word in train_unique_words:
            index = np.where(train_unique_words == word)[0]
            x[index] = 1
    test_x = np.vstack((test_x, x))

for title in real[real_test_i]:
    x = np.zeros(dim_x)
    for word in title.split(' '):
        if word in train_unique_words:
            index = np.where(train_unique_words == word)[0]
            x[index] = 1
    test_x = np.vstack((test_x, x))  
test_x = test_x[1:]

# construct labels
test_y =np.vstack((np.tile(np.array(0), (fake_test_i.size, 1)), \
                   np.tile(np.array(1), (real_test_i.size, 1))))

    
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

np.random.seed(SEED)
train_idx = np.random.permutation(range(train_x.shape[0]))
x_train = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
y_classes_train = Variable(torch.from_numpy(train_y[train_idx]), requires_grad=False).type(dtype_float)

np.random.seed(SEED)
valid_idx = np.random.permutation(range(valid_x.shape[0]))
x_valid = Variable(torch.from_numpy(valid_x[valid_idx]), requires_grad=False).type(dtype_float)
y_classes_valid = Variable(torch.from_numpy(valid_y[valid_idx]), requires_grad=False).type(dtype_float)

np.random.seed(SEED)
test_idx = np.random.permutation(range(test_x.shape[0]))
x_test = Variable(torch.from_numpy(test_x[test_idx]), requires_grad=False).type(dtype_float)
y_classes_test = Variable(torch.from_numpy(test_y[test_idx]), requires_grad=False).type(dtype_float)


def LR_accuracy(x, y):
    correct_count = 0
    p_data = model(x).data.numpy()
    y_data = y.data.numpy()
    for i in range(len(x)):
        if y_data[i] == 1:
            if p_data[i] >= 0.5:
                correct_count += 1
        elif y_data[i] == 0:
            if p_data[i] < 0.5:
                correct_count += 1
    return correct_count/float(len(x))


# run from here for training model
model = torch.nn.Sequential(
            torch.nn.Linear(dim_x, 1),
            torch.nn.Sigmoid(),
        )
    
loss_fn = torch.nn.BCELoss() 

train_acc = list()
valid_acc = list()
valid_cost = list()
epoch_list = list()    

learning_rate = 1e-4
EPS = 1e-6
max_iter = 50000
freq = 100
count = 3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)

# Xavier Initialization
np.random.seed(SEED) 
W = np.random.normal(0, 2/float(dim_x + dim_out), size=(dim_out, dim_x))

np.random.seed(SEED) 
b = np.random.normal(0, 2/float(dim_x + dim_out), size=dim_out)

model[0].weight.data = torch.from_numpy(W).type(dtype_float)
model[0].bias.data = torch.from_numpy(b).type(dtype_float)

W = model[0].weight.data.numpy()
prev_W = W-100*EPS
    
epoch = 0
while np.linalg.norm(W - prev_W) > EPS and epoch < max_iter \
    and np.count_nonzero(np.diff(valid_cost)[-count:] >= 0) < count:
    prev_W = W.copy()

    y_pred_train = model(x_train)
    loss_train = loss_fn(y_pred_train, y_classes_train)

    y_pred_valid = model(x_valid)
    loss_valid = loss_fn(y_pred_valid, y_classes_valid)
    
    model.zero_grad()  
    loss_train.backward()    
    optimizer.step()  

    W = model[0].weight.data.numpy()
    
    if epoch % freq == 0:
        print "Epoch ", epoch
        
        # report costs
        t_acc = LR_accuracy(x_train, y_classes_train)
        v_acc = LR_accuracy(x_valid, y_classes_valid)
        
        v_cost = loss_valid.data.numpy()[0]

        train_acc.append(t_acc)
        valid_acc.append(v_acc)
        
        valid_cost.append(v_cost)

        print '\ntraining accuracy: %.7f\nvalidation accuracy: %.7f\n'% (t_acc, v_acc)
        
        epoch_list.append(epoch)
        
    epoch += 1


# final performance
print ('\n--------- Final Performance ---------')
print '\ntraining accuracy: %.2f%%\nvalidation accuracy: %.2f%%\ntest accuracy: %.2f%%\n' \
    % (LR_accuracy(x_train, y_classes_train)*100, \
       LR_accuracy(x_valid, y_classes_valid)*100, \
       LR_accuracy(x_test, y_classes_test)*100)

# plot learning curve
plt.plot(epoch_list, train_acc, color='blue')
plt.plot(epoch_list, valid_acc, color='green')
plt.title('Learning Curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

############ Part 6 - Weights in LR ############ 

weights = model[0].weight.data.numpy().reshape((-1,))
sorted_wi = np.argsort(weights)[:][::-1]

# indices of top 10 largest negative weights and words associated to them
for index in sorted_wi[:10]:
    print index, weights[index], train_unique_words[index]

# indices of top 10 lowest positive weights and words associated to them
for index in sorted_wi[-10:]:
    print index, weights[index], train_unique_words[index]

# nonstop words
nonstop_index = np.array(0)
for index in sorted_wi:
    if train_unique_words[index] not in ENGLISH_STOP_WORDS:
        nonstop_index = np.append(nonstop_index, index)     
nonstop_index = nonstop_index[1:]
    
# indices of top 10 largest positive weights and words associated to them
for index in nonstop_index[:10]:
    print index, weights[index], train_unique_words[index]
    
# indices of top 10 lowest negative weights and words associated to them
for index in nonstop_index[-10:]:
    print index, weights[index], train_unique_words[index]


########### Part 7 - Decision Tree ###########

# 7 a)

features = list(train_unique_words)

def DT_accuracy(x, y):
    y_pred = np.array(c.predict(x))
    y_actual = y.T.flatten()
    return 1 - np.count_nonzero(y_actual - y_pred)/float(y_pred.shape[0])


depth_list = []
DT_train_acc = [] 
DT_valid_acc = []
best_v_acc = 0

# criterion max_leaf_nodes
for depth in range(2, 100):
    c = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 100, \
                               max_depth = depth, random_state = SEED)
    dt = c.fit(train_x[train_idx], train_y[train_idx])
    
    DT_t_acc = DT_accuracy(train_x[train_idx], train_y[train_idx])
    DT_v_acc = DT_accuracy(valid_x[valid_idx], valid_y[valid_idx])
    
    if DT_v_acc > best_v_acc:
        best_v_acc = DT_v_acc
        best_depth = depth
    
    depth_list.append(depth)
    DT_train_acc.append(DT_t_acc)
    DT_valid_acc.append(DT_v_acc)
    
    print '\nDepth ', depth
    print '\ntraining accuracy: %.2f\nvalidation accuracy: %.2f\ntest accuracy: %.2f\n' \
        %  (DT_t_acc, \
            DT_v_acc, \
            DT_accuracy(test_x[test_idx], test_y[test_idx]))
    
# plot performance vs depth
plt.plot(depth_list, DT_train_acc, color='blue')
plt.plot(depth_list, DT_valid_acc, color='green')
plt.title('Performance vs. Depth')
plt.ylabel('accuracy')
plt.xlabel('depth')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()   


# choose the best-performing tree
c = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 100, \
                           max_depth = best_depth, random_state = SEED)
dt = c.fit(train_x[train_idx], train_y[train_idx])  

print ('\n--------- Best Performance ---------')
print '\ntraining accuracy: %.2f%%\nvalidation accuracy: %.2f%%\ntest accuracy: %.2f%%\n' \
    % (DT_accuracy(train_x[train_idx], train_y[train_idx])*100, \
       DT_accuracy(valid_x[valid_idx], valid_y[valid_idx])*100, \
       DT_accuracy(test_x[test_idx], test_y[test_idx])*100)
    
# 7 b) visualization
def show_tree(tree, features, depth, path):
    f = StringIO.StringIO()
    export_graphviz(tree, out_file = f, max_depth = depth, rounded = True,
                    feature_names = features, class_names = ['fake', 'real'])
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = misc.imread(path)
    plt.imshow(img)

show_tree(dt, features, 2, 'dt.png')


######### Part 8 - Calculating Mutual Information ########

# entropy calculation given number of fake/real news healines in a node
def H(num_fake, num_real):
    num = float(num_fake + num_real)
    p_fake = num_fake/num
    p_real = num_real/num
    return -(p_fake*math.log(p_fake, 2) 
             + p_real*math.log(p_real, 2))
    
# I(Y; xj) = H(Y) - H(Y | xj) 
def mutual_info(hy, p_word, p_noword, hy_word, hy_noword):
    hy_xj = p_word * hy_word + p_noword * hy_noword
    return hy - hy_xj

# 8 a)  mutual information when splitting on word "trumps"
print '\nmutual information when splitting on "trump":'
print mutual_info(H(909, 1378), 1378/2287., 909/2287., H(1, 153), H(908, 1225))

# 8 b)  mutual information when splitting on word "hillary"
word_i = np.argwhere(train_unique_words == "hillary")

train_set = train_x[train_idx]
labels = train_y[train_idx]

fake_i = np.argwhere(train_set[:,word_i] == 0)[:,0]
real_i = np.argwhere(train_set[:,word_i] != 0)[:,0]

# when the word not exists
num_real_0 = np.count_nonzero(labels[real_i])
num_fake_0 = len(labels[real_i]) - num_real_0

# when the word exists
num_real_1 = np.count_nonzero(labels[fake_i])
num_fake_1 = len(labels[fake_i]) - num_real_1

print '\nmutual information when splitting on "hillary":'
print mutual_info(H(909, 1378), 1378/2287., 909/2287., \
            H(num_fake_0, num_real_0), H(num_fake_1, num_real_1))





