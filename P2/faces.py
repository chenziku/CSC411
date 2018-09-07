#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:16:53 2018

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
from torch.autograd import Variable
import torch


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
## download and preprocess images of Peri Gilpin
#if not os.path.exists('gilpin'):
#    os.makedirs('gilpin')
#if not os.path.exists('gilpin_crop'):
#    os.makedirs('gilpin_crop')
#
#testfile = urllib.URLopener()   
#
#i = 0
#name = "gilpin"
#for line in open("gilpin.txt"):
#    filetype = line.split()[4].split('.')[-1]
#    if "Peri Gilpin" in line:
#        filename_raw = (name+str(i)+'_raw'+'.'+filetype).lower()
#        timeout(testfile.retrieve, (line.split()[4], "gilpin/"+filename_raw), {}, 120)
#        try:
#            im = scipy.misc.imread("gilpin/"+filename_raw)
#            filename =name+str(i)+'.'+filetype
#            x1 = int(line.split()[5].split(',')[0])
#            y1 = int(line.split()[5].split(',')[1])
#            x2 = int(line.split()[5].split(',')[2])
#            y2 = int(line.split()[5].split(',')[3])
#            im = imresize(im[y1:y2, x1:x2], (32, 32))
#            scipy.misc.imsave("gilpin_crop/"+filename.split('_')[0], im)
#            print filename     
#        except IOError:
#            pass
#        i += 1
#
## download and preprocess images of other actors
#if not os.path.exists('uncropped'):
#    os.makedirs('uncropped')
#
#if not os.path.exists('data'):
#    os.makedirs('data')
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
#            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename_raw), {}, 120)
#            if hashlib.sha256(open("uncropped/"+filename_raw, "rb").read()).hexdigest() == line.split()[-1]:
#                try:
#                    im = scipy.misc.imread("uncropped/"+filename_raw)
#                    filename =name+str(i)+'.'+filetype
#                    x1 = int(line.split()[5].split(',')[0])
#                    y1 = int(line.split()[5].split(',')[1])
#                    x2 = int(line.split()[5].split(',')[2])
#                    y2 = int(line.split()[5].split(',')[3])
#                    im = imresize(im[y1:y2, x1:x2], (32, 32))
#                    scipy.misc.imsave("data/"+filename.split('_')[0], im)
#                    print filename     
#                except IOError:
#                    pass
#                i += 1       
#
#
#for a in actors:
#    name = a.split()[1].lower()
#    i = 0
#    for line in open("facescrub_actors.txt"):
#        filetype = line.split()[4].split('.')[-1]
#        if a in line:
#            filename_raw = (name+str(i)+'_raw'+'.'+filetype).lower()
#            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename_raw), {}, 120)
#            if hashlib.sha256(open("uncropped/"+filename_raw, "rb").read()).hexdigest() == line.split()[-1]:
#                try:
#                    im = scipy.misc.imread("uncropped/"+filename_raw)
#                    filename =name+str(i)+'.'+filetype
#                    x1 = int(line.split()[5].split(',')[0])
#                    y1 = int(line.split()[5].split(',')[1])
#                    x2 = int(line.split()[5].split(',')[2])
#                    y2 = int(line.split()[5].split(',')[3])
#                    im = imresize(im[y1:y2, x1:x2], (32, 32))
#                    scipy.misc.imsave("data/"+filename.split('_')[0], im)
#                    print filename     
#                except IOError:
#                    pass
#                i += 1



############ Construct Training, Validation and Test Sets ############

dim_x = 32*32*3
dim_h = 21
dim_out = 6
SEED = 2018

act_lastname = [name.split(' ')[1].lower() for name in act]

for files in os.walk("data/"):  
    files = files[-1]

# for reproducibility
files.sort()
np.random.seed(SEED)
np.random.shuffle(files)

bracco = np.zeros(dim_x)
gilpin = np.zeros(dim_x)
harmon = np.zeros(dim_x)
baldwin = np.zeros(dim_x)
hader = np.zeros(dim_x)
carell = np.zeros(dim_x)

for filename in files:
    name = lastname(filename)
    ft = filetype(filename)
    if ft == "png":
        im = scipy.misc.imread("data/" + filename)[:,:,:3].flatten()
    elif ft in ['jpg', 'jpeg']:
        im = scipy.misc.imread("data/" + filename).flatten()
    if im.shape == (32*32, ):
        im = np.tile(im.reshape(32,32, 1), 3).flatten()
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
faces['train1'] = gilpin[:58, :]
faces['valid1'] = gilpin[58:72, :]
faces['test1'] = gilpin[72:87, :]

# other actors

split1 = 79
split2 = 99
split3 = 119

faces['train0'] = bracco[:split1, :]
faces['train2'] = harmon[:split1, :]
faces['train3'] = baldwin[:split1, :]
faces['train4'] = hader[:split1, :]
faces['train5'] = carell[:split1, :]

faces['valid0'] = bracco[split1:split2, :]
faces['valid2'] = harmon[split1:split2, :]
faces['valid3'] = baldwin[split1:split2, :]
faces['valid4'] = hader[split1:split2, :]
faces['valid5'] = carell[split1:split2, :]

faces['test0'] = bracco[split2:split3, :]
faces['test2'] = harmon[split2:split3, :]
faces['test3'] = baldwin[split2:split3, :]
faces['test4'] = hader[split2:split3, :]
faces['test5'] = carell[split2:split3, :]


def get_train(M):
    batch_xs = np.zeros((0, dim_x))
    batch_y_s = np.zeros((0, dim_out))
    
    train_k =  ["train"+str(i) for i in range(dim_out)]
    for k in range(dim_out):
        batch_xs = np.vstack((batch_xs, ((np.array(M[train_k[k]])[:]))))
        one_hot = np.zeros(dim_out)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def get_valid(M):
    batch_xs = np.zeros((0, dim_x))
    batch_y_s = np.zeros( (0, dim_out))
    
    valid_k =  ["valid"+str(i) for i in range(dim_out)]
    for k in range(dim_out):
        batch_xs = np.vstack((batch_xs, ((np.array(M[valid_k[k]])[:]))  ))
        one_hot = np.zeros(dim_out)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[valid_k[k]]), 1))   ))
    return batch_xs, batch_y_s 

def get_test(M):
    batch_xs = np.zeros((0, dim_x))
    batch_y_s = np.zeros( (0, dim_out))
    
    test_k =  ["test"+str(i) for i in range(dim_out)]
    for k in range(dim_out):
        batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:]))  ))
        one_hot = np.zeros(dim_out)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s       

def accuracy(x, y):
    correct_count = 0
    for i in range(len(x)):
        if np.argmax(model(x).data.numpy()[i]) == y.data.numpy()[i]:
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
# define layers and loss function (run below to repeat training)
model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)

loss_fn = torch.nn.CrossEntropyLoss()

train_costs = list()
valid_costs = list()
epoch_list = list()    

learning_rate = 1e-3
EPS = 1e-6
max_iter = 50000
freq = 10
count = 3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# initializing the weights and biases
np.random.seed(SEED) 
W0 = np.random.normal(0, 1e-4, size=(dim_h, dim_x))

np.random.seed(SEED) 
b0 = np.random.normal(0, 1e-4, size=dim_h)

np.random.seed(SEED) 
W1 = np.random.normal(0, 1e-4, size=(dim_out, dim_h))

np.random.seed(SEED) 
b1 = np.random.normal(0, 1e-4, size=dim_out)

model[0].weight.data = torch.from_numpy(W0).type(dtype_float)
model[0].bias.data = torch.from_numpy(b0).type(dtype_float)
model[2].weight.data = torch.from_numpy(W1).type(dtype_float)
model[2].bias.data = torch.from_numpy(b1).type(dtype_float)

W = model[0].weight.data.numpy()
prev_W = W-100*EPS
    
epoch = 0
while np.linalg.norm(W - prev_W) > EPS and epoch < max_iter \
    and np.count_nonzero(np.diff(valid_costs)[-count:]>0) < count:
        
    
    prev_W = W.copy()

    
    y_pred_train = model(x_train)
    loss_train = loss_fn(y_pred_train, y_classes_train)

    y_pred_valid = model(x_valid)
    loss_valid = loss_fn(y_pred_valid, y_classes_valid)
    
    model.zero_grad()  # Zero out the previous gradient computation
    loss_train.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step
 
    W = model[0].weight.data.numpy()
    
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



################## Part 9 ##################

# pick an image of Angie Harmon
harmon = Variable(torch.from_numpy(faces['train2'][0].flatten()), requires_grad=False).type(dtype_float)

# make sure it is classified correctly by the trained model
np.argmax(model(harmon).data.numpy()) == 2

# determine which hidden units has the largest value for the particular image
h = np.argmax(model[0](harmon).data.numpy())

# visualize the weights associated with the unit
w = model[0].weight.data.numpy()[h, :]

f, axarr = plt.subplots(1, 3)

axarr[0].imshow(w.reshape(32,32,3)[:,:,0], cmap=plt.cm.coolwarm)
axarr[0].axis('off')

axarr[1].imshow(w.reshape(32,32,3)[:,:,1], cmap=plt.cm.coolwarm)
axarr[1].axis('off')

axarr[2].imshow(w.reshape(32,32,3)[:,:,2], cmap=plt.cm.coolwarm)
axarr[2].axis('off')

plt.show()


# pick an image of Alec Baldwin
baldwin = Variable(torch.from_numpy(faces['train4'][50].flatten()), requires_grad=False).type(dtype_float)

# make sure it is classified correctly by the trained model
np.argmax(model(baldwin).data.numpy()) == 3

# determine which hidden units has the largest value for the particular image
h = np.argmax(model[0](baldwin).data.numpy())

# visualize the weights associated with the unit
w = model[0].weight.data.numpy()[h, :]

f, axarr = plt.subplots(1, 3)

axarr[0].imshow(w.reshape(32,32,3)[:,:,0], cmap=plt.cm.coolwarm)
axarr[0].axis('off')

axarr[1].imshow(w.reshape(32,32,3)[:,:,1], cmap=plt.cm.coolwarm)
axarr[1].axis('off')

axarr[2].imshow(w.reshape(32,32,3)[:,:,2], cmap=plt.cm.coolwarm)
axarr[2].axis('off')

plt.show()


