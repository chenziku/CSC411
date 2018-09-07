#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:23:17 2018

@author: zikunchen
"""

import re
import numpy as np
from matplotlib import cm
import scipy
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from scipy.misc import imshow
import os
import urllib
import hashlib
import matplotlib.image as mpimg
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch

SEED = 2018
dim_out = 6

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()   

def lastname(filename):
    return re.split("[^a-z]*", filename.split('.')[0])[0]
def filetype(filename):
    return filename.split('.')[-1].lower()


act = list()
for line in open("faces_subset.txt").readlines():
    act.append(line.split('\n')[0])
actresses = act[:3]
actors = act[3:]


############### DOWNLOAD & PREPROCCESSING ###############
## do not need to be ran again since the zip files 
## of the processed data set is uploaded
#
#testfile = urllib.URLopener()  
#
## download and preprocess images of Peri Gilpin
#if not os.path.exists('gilpin_deep'):
#    os.makedirs('gilpin_deep')
#if not os.path.exists('gilpin_crop_deep'):
#    os.makedirs('gilpin_crop_deep')
#
#i = 0
#name = "gilpin"
#for line in open("gilpin.txt"):
#    filetype = line.split()[4].split('.')[-1]
#    if "Peri Gilpin" in line:
#        filename_raw = (name+str(i)+'_raw'+'.'+filetype).lower()
#        timeout(testfile.retrieve, (line.split()[4], "gilpin_deep/"+filename_raw), {}, 120)
#        try:
#            im = scipy.misc.imread("gilpin_deep/"+filename_raw)
#            filename =name+str(i)+'.'+filetype
#            x1 = int(line.split()[5].split(',')[0])
#            y1 = int(line.split()[5].split(',')[1])
#            x2 = int(line.split()[5].split(',')[2])
#            y2 = int(line.split()[5].split(',')[3])
#            im = imresize(im[y1:y2, x1:x2], (227, 227))
#            scipy.misc.imsave("gilpin_crop_deep/"+filename.split('_')[0], im)
#            print filename     
#        except IOError:
#            pass
#        i += 1
#
## download and preprocess images of other actors
#if not os.path.exists('deep_uncropped'):
#    os.makedirs('deep_uncropped')
#if not os.path.exists('deep'):
#    os.makedirs('deep')
#
#
#for a in actresses:
#    name = a.split()[1].lower()
#    i = 0
#    for line in open("facescrub_actresses.txt"):
#        filetype = line.split()[4].split('.')[-1]
#        if filetype == "jpeg":
#            filetype = "jpg"
#        if a in line and name  != "gilpin":
#            filename_raw = (name+str(i)+'_raw'+'.'+filetype).lower()
#            timeout(testfile.retrieve, (line.split()[4], "deep_uncropped/"+filename_raw), {}, 120)
#            if hashlib.sha256(open("deep_uncropped/"+filename_raw, "rb").read()).hexdigest() == line.split()[-1]:
#                try:
#                    im = scipy.misc.imread("deep_uncropped/"+filename_raw)
#                    filename =name+str(i)+'.'+filetype
#                    x1 = int(line.split()[5].split(',')[0])
#                    y1 = int(line.split()[5].split(',')[1])
#                    x2 = int(line.split()[5].split(',')[2])
#                    y2 = int(line.split()[5].split(',')[3])
#                    im = imresize(im[y1:y2, x1:x2], (227, 227))
#                    scipy.misc.imsave("deep/"+filename.split('_')[0], im)
#                    print filename     
#                except IOError:
#                    pass
#            i += 1    
#            
#for a in actors:
#    name = a.split()[1].lower()
#    i = 0
#    for line in open("facescrub_actors.txt"):
#        filetype = line.split()[4].split('.')[-1]
#        if filetype == "jpeg":
#            filetype = "jpg"
#        if a in line:
#            filename_raw = (name+str(i)+'_raw'+'.'+filetype).lower()
#            timeout(testfile.retrieve, (line.split()[4], "deep_uncropped/"+filename_raw), {}, 120)
#            if hashlib.sha256(open("deep_uncropped/"+filename_raw, "rb").read()).hexdigest() == line.split()[-1]:
#                try:
#                    im = scipy.misc.imread("deep_uncropped/"+filename_raw)
#                    filename =name+str(i)+'.'+filetype
#                    x1 = int(line.split()[5].split(',')[0])
#                    y1 = int(line.split()[5].split(',')[1])
#                    x2 = int(line.split()[5].split(',')[2])
#                    y2 = int(line.split()[5].split(',')[3])
#                    im = imresize(im[y1:y2, x1:x2], (227, 227))
#                    scipy.misc.imsave("deep/"+filename.split('_')[0], im)
#                    print filename     
#                except IOError:
#                    pass
#            i += 1   

    

############ Construct Training, Validation and Test Sets ############

act_lastname = [name.split(' ')[1].lower() for name in act]


for files in os.walk("deep/"):  
    files = files[-1]
    
# for reproducibility
files.sort()
np.random.seed(SEED)
np.random.shuffle(files)

bracco = np.zeros((1, 3, 227, 227))
gilpin = np.zeros((1, 3, 227, 227))
harmon = np.zeros((1, 3, 227, 227))
baldwin = np.zeros((1, 3, 227, 227))
hader = np.zeros((1, 3, 227, 227))
carell = np.zeros((1, 3, 227, 227))
    
for filename in files:
    name = lastname(filename)
    ft = filetype(filename)
    if ft == "png":
        im = scipy.misc.imread("deep/" + filename)[:,:,:3]
    elif ft in ['jpg', 'jpeg']:
        im = scipy.misc.imread("deep/" + filename)
    if im.shape == (227, 227):
        im = np.tile(im.reshape(227,227, 1), 3)
    im = im - np.mean(im.flatten())
    im = im/np.max(np.abs(im.flatten()))
    im = np.rollaxis(im, -1).astype(np.float32)
    im = im.reshape(1, 3, 227, 227)
    if name == act_lastname[0]:
        bracco = np.vstack((im,bracco))
    elif name == act_lastname[1]:
        gilpin = np.vstack((im,gilpin))
    elif name == act_lastname[2]:
        harmon = np.vstack((im,harmon))
    elif name == act_lastname[3]:
        baldwin = np.vstack((im,baldwin))
    elif name == act_lastname[4]:
        hader = np.vstack((im,hader))
    elif name == act_lastname[5]:
        carell = np.vstack((im,carell))


bracco = bracco[:-1, :]/255.
gilpin = gilpin[:-1, :]/255.
harmon = harmon[:-1, :]/255.
baldwin = baldwin[:-1, :]/255.
hader = hader[:-1, :]/255.
carell = carell[:-1, :]/255.


faces = {}

# proportionally decrease images for Gilpin in validation set and test set
faces['train1'] = gilpin[:58, :, :, :]
faces['valid1'] = gilpin[58:73, :, :, :]
faces['test1'] = gilpin[73:88, :, :, :]

# other actors
split1 = 78
split2 = 98
split3 = 118

faces['train0'] = bracco[:split1, :, :, :]
faces['train2'] = harmon[:split1, :, :, :]
faces['train3'] = baldwin[:split1, :, :, :]
faces['train4'] = hader[:split1, :, :, :]
faces['train5'] = carell[:split1, :, :, :]

faces['valid0'] = bracco[split1:split2, :, :, :]
faces['valid2'] = harmon[split1:split2, :, :, :]
faces['valid3'] = baldwin[split1:split2, :, :, :]
faces['valid4'] = hader[split1:split2, :, :, :]
faces['valid5'] = carell[split1:split2, :, :, :]

faces['test0'] = bracco[split2:split3, :, :, :]
faces['test2'] = harmon[split2:split3, :, :, :]
faces['test3'] = baldwin[split2:split3, :, :, :]
faces['test4'] = hader[split2:split3, :, :, :]
faces['test5'] = carell[split2:split3, :, :, :]


def get_train(M):
    batch_xs = np.zeros((0, 3, 227, 227))
    batch_y_s = np.zeros((0, dim_out))
    train_k =  ["train"+str(i) for i in range(dim_out)]
    for k in range(dim_out):
        batch_xs = np.vstack((batch_xs, ((np.array(M[train_k[k]])[:]))  ))
        one_hot = np.zeros(dim_out)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[train_k[k]]), 1))))
    
    return batch_xs, batch_y_s

def get_valid(M):
    batch_xs = np.zeros((0, 3, 227, 227))
    batch_y_s = np.zeros((0, dim_out))
    valid_k =  ["valid"+str(i) for i in range(dim_out)]
    for k in range(dim_out):
        batch_xs = np.vstack((batch_xs, ((np.array(M[valid_k[k]])[:]))  ))
        one_hot = np.zeros(dim_out)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[valid_k[k]]), 1))))
    
    return batch_xs, batch_y_s

def get_test(M):
    batch_xs = np.zeros((0, 3, 227, 227))
    batch_y_s = np.zeros((0, dim_out))
    test_k =  ["test"+str(i) for i in range(dim_out)]
    for k in range(dim_out):
        batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:]))  ))
        one_hot = np.zeros(dim_out)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[test_k[k]]), 1))))
    
    return batch_xs, batch_y_s   

def accuracy(x, y):
    correct_count = 0
    x_extract = model_full(x).data.numpy()
    for i in range(len(x)):
        if np.argmax(x_extract[i]) == y.data.numpy()[i]:
            correct_count += 1
    return correct_count/float(len(x))


train_x, train_y = get_train(faces)
valid_x, valid_y = get_valid(faces)
test_x, test_y = get_test(faces)


dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

np.random.seed(SEED)
train_idx = np.random.permutation(range(train_x.shape[0]))
x_train = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
y_classes_train = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)

np.random.seed(SEED)
valid_idx = np.random.permutation(range(valid_x.shape[0]))
x_valid = Variable(torch.from_numpy(valid_x[valid_idx]), requires_grad=False).type(dtype_float)
y_classes_valid = Variable(torch.from_numpy(np.argmax(valid_y[valid_idx], 1)), requires_grad=False).type(dtype_long)

np.random.seed(SEED)
test_idx = np.random.permutation(range(test_x.shape[0]))
x_test = Variable(torch.from_numpy(test_x[test_idx]), requires_grad=False).type(dtype_float)
y_classes_test = Variable(torch.from_numpy(np.argmax(test_y[test_idx], 1)), requires_grad=False).type(dtype_long)

############ Construct Neural Network and Training ############

# layers and weights from AlexNet
model_alex = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )   
# load and fix weights 
an_builtin = torchvision.models.alexnet(pretrained=True)

extract_weight_i = [0, 3, 6, 8]
for i in extract_weight_i:
    model_alex[i].weight = an_builtin.features[i].weight
    model_alex[i].bias = an_builtin.features[i].bias
    for param in model_alex[i].parameters():
        param.requires_grad = False  

def extract(x):
    x = model_alex(x)
    x = x.view(x.size(0), 256 * 13 * 13)
    return x


# feed training sets through AlexNet to form new training data
# so that we can extract features of images
x_train = extract(x_train)
x_valid = extract(x_valid)
x_test = extract(x_test)



# fully-connected layer to train (run below to repeat training)
dim_h = 50
model_full = nn.Sequential(
                nn.Linear(256 * 13 * 13 , dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_out),
            )

loss_fn = torch.nn.CrossEntropyLoss()

# list of costs for ploting the learning curves
train_costs = list()
valid_costs = list()
epoch_list = list()    

# hyperparameters
learning_rate = 1e-5
EPS = 1e-5
max_iter = 1000000
freq = 50
count = 5
optimizer = torch.optim.Adam(model_full.parameters(), lr=learning_rate)

# initializing the weights and biases
np.random.seed(SEED) 
W0 = np.random.normal(0, 1e-5, size=(dim_h, 256 * 13 * 13))

np.random.seed(SEED) 
b0 = np.random.normal(0, 1e-5, size=dim_h)

np.random.seed(SEED) 
W1 = np.random.normal(0, 1e-5, size=(dim_out, dim_h))

np.random.seed(SEED) 
b1 = np.random.normal(0, 1e-5, size=dim_out)

model_full[0].weight.data = torch.from_numpy(W0).type(dtype_float)
model_full[0].bias.data = torch.from_numpy(b0).type(dtype_float)
model_full[2].weight.data = torch.from_numpy(W1).type(dtype_float)
model_full[2].bias.data = torch.from_numpy(b1).type(dtype_float)


W = model_full[0].weight.data.numpy()
prev_W = W-100*EPS
    
epoch = 0

while  np.linalg.norm(W - prev_W) > EPS and epoch < max_iter \
    and np.count_nonzero(np.diff(valid_costs)[-count:]>0) < count:
    
    prev_W = W.copy()
    
    y_pred_train = model_full(x_train)
    loss_train = loss_fn(y_pred_train, y_classes_train)
    
    y_pred_valid = model_full(x_valid)
    loss_valid = loss_fn(y_pred_valid, y_classes_valid)
    
    
    model_full.zero_grad()  # Zero out the previous gradient computation
    loss_train.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step
 
    W = model_full[0].weight.data.numpy()
    
    if epoch % freq == 0:
        print "Epoch ", epoch

        # report costs
        t_cost = loss_train.data.numpy()[0]
        v_cost = loss_valid.data.numpy()[0]

        train_costs.append(t_cost)
        valid_costs.append(v_cost)

        print '\ntraining cost: %.7f\nvalidation cost: %.7f\n'% (t_cost, v_cost)
        
        epoch_list.append(epoch)
    epoch += 1
    
    
# final performance
print ('\n--------- Final Performance ---------')
print '\ntraining accuracy: %.2f%%\nvalidation accuracy: %.2f%%\ntest accuracy: %.2f%%\n' \
    % (accuracy(x_train, y_classes_train)*100, accuracy(x_valid, y_classes_valid)*100, accuracy(x_test, y_classes_test)*100)

    
# plot learning curve
plt.plot(epoch_list, train_costs, color='blue')
plt.plot(epoch_list, valid_costs, color='green')
plt.title('Learning Curve')
plt.ylabel('cost')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
plt.figure()

