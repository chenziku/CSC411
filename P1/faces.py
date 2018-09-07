#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:16:53 2018

@author: zikunchen
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from scipy.misc import imshow
import os
import urllib


##########################  Downloading and Cleaning  ##########################


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

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
    return filename.split('.')[-1]


## creating folders
#if not os.path.exists('uncropped'):
#    os.makedirs('uncropped')
#
#if not os.path.exists('data'):
#    os.makedirs('data')

 
act = list()
for line in open("faces_subset.txt").readlines():
    act.append(line.split('\n')[0])
actresses = act[:3]
actors = act[3:]

#
#for a in actresses:
#    # last name lower case
#    name = a.split()[1].lower()
#    i = 0
#    for line in open("facescrub_actresses.txt"):
#        filetype = line.split()[4].split('.')[-1]
#        if a in line:
#            filename_raw = (name+str(i)+'_raw'+'.'+filetype).lower()
#            #A version without timeout (uncomment in case you need to 
#            #unsupress exceptions, which timeout() does)
#            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
#            #timeout is used to stop downloading images which take too long to download
#            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename_raw), {}, 35)
#            if not os.path.isfile("uncropped/"+filename_raw):
#                continue
#            try:
#                im = plt.imread("uncropped/"+filename_raw)
#                filename =name+str(i)+'.'+filetype
#                if len(im.shape) == 3: 
#                    im = rgb2gray(im)
#                x1 = int(line.split()[5].split(',')[0])
#                y1 = int(line.split()[5].split(',')[1])
#                x2 = int(line.split()[5].split(',')[2])
#                y2 = int(line.split()[5].split(',')[3])
#                im = imresize(im[y1:y2, x1:x2], (32, 32))
#                plt.imsave("data/"+filename.split('_')[0], im, cmap = plt.cm.gray)
#            except IOError:
#                pass
#
#            print filename
#            i += 1
#
#for a in actors:
#    # last name lower case
#    name = a.split()[1].lower()
#    i = 0
#    for line in open("facescrub_actors.txt"):
#        filetype = line.split()[4].split('.')[-1]
#        if a in line:
#            filename_raw = (name+str(i)+'_raw'+'.'+filetype).lower()
#            #A version without timeout (uncomment in case you need to 
#            #unsupress exceptions, which timeout() does)
#            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
#            #timeout is used to stop downloading images which take too long to download
#            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename_raw), {}, 35)
#            if not os.path.isfile("uncropped/"+filename_raw.lower()):
#                continue
#            try:
#                im = imread("uncropped/"+filename_raw)
#                filename =name+str(i)+'.'+filetype
#                if len(im.shape) == 3: 
#                    im = rgb2gray(im)
#                x1 = int(line.split()[5].split(',')[0])
#                y1 = int(line.split()[5].split(',')[1])
#                x2 = int(line.split()[5].split(',')[2])
#                y2 = int(line.split()[5].split(',')[3])
#                im = imresize(im[y1:y2, x1:x2], (32, 32))
#                plt.imsave("data/"+filename.split('_')[0], im, cmap = plt.cm.gray)
#            except IOError:
#                pass
#
#            print filename
#            i += 1
#
#    
## remove expired and mis-cropped images
#for line in open("badimages.txt"):
#    try:
#        os.remove('data/'+line.split('\n')[0])
#    except OSError:
#        pass

##########################  PART 2  ##########################
    

# get file names for different actors
act_lastname = [name.split(' ')[1].lower() for name in act]
harmon = list()
bracco = list()
gilpin = list()
baldwin = list()
hader = list()
carell = list()
for files in os.walk("data/"):  
    files = files[-1]
for filename in files:
    name = lastname(filename)
    if name == act_lastname[0]:
        harmon.append(filename)
    elif name == act_lastname[1]:
        bracco.append(filename)
    elif name == act_lastname[2]:
        gilpin.append(filename)
    elif name == act_lastname[3]:
        baldwin.append(filename)
    elif name == act_lastname[4]:
        hader.append(filename)
    elif name == act_lastname[5]:
        carell.append(filename)


# construct lists of training, validation and test files
SEED = 2018

baldwin.sort()
np.random.seed(SEED)
np.random.shuffle(baldwin)
baldwin_training = baldwin[0:70]
baldwin_validation = baldwin[70:80]
baldwin_test = baldwin[80:90]

carell.sort()
np.random.seed(SEED)
np.random.shuffle(carell)
carell_training = carell[0:70]
carell_validation = carell[70:80]
carell_test = carell[80:90]

hader.sort()
np.random.seed(SEED)
np.random.shuffle(hader)
hader_training = hader[0:70]
hader_validation = hader[70:80]
hader_test = hader[80:90]

gilpin.sort()
np.random.seed(SEED)
np.random.shuffle(gilpin)
gilpin_training = gilpin[0:70]
gilpin_validation = gilpin[70:80]
gilpin_test = gilpin[80:90]

bracco.sort()
np.random.seed(SEED)
np.random.shuffle(bracco)
bracco_training = bracco[0:70]
bracco_validation = bracco[70:80]
bracco_test = bracco[80:90]

harmon.sort()
np.random.seed(SEED)
np.random.shuffle(harmon)
harmon_training = harmon[0:70]
harmon_validation = harmon[70:80]
harmon_test = harmon[80:90]


training_files = baldwin_training + carell_training
np.random.seed(SEED)
np.random.shuffle(training_files)

validation_files = baldwin_validation + carell_validation
np.random.seed(SEED)
np.random.shuffle(validation_files)

test_files = baldwin_test + carell_test
np.random.seed(SEED)
np.random.shuffle(test_files)



##########################  PART 3  ##########################

def f(x, y, theta):
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return sum((y - np.dot(theta.T,x)) ** 2)
    
def df(x, y, theta):
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return -2*np.sum((y-np.dot(theta.T, x))*x, 1)

def accuracy(x, y, t):
    M = float(x.shape[1])
    return np.divide(M-(np.count_nonzero((np.dot(t.T, x)>0.5)-y)) ,M)

def check_gradient(x, y, theta, i, h=1e-5):
    deltai = np.zeros(theta.shape)
    deltai[i] = h
    print 'index %1d:'%(i)
    print 'finite-difference approximation: %.7f\ndf(x, y, theta): %.7f\n'\
        % ((f(x, y, theta+deltai) - f(x, y, theta))/h, df(x, y, theta)[i])


def train(training_files, validation_files, test_files, init_t, alpha = 1e-6, EPS=1e-5, freq=500, max_iter=300000):
    M = len(training_files)
    N = 32*32
    
    # creating labels for training set  
    y_train = np.empty([len(training_files)])
    for i in range(len(training_files)):
        if lastname(training_files[i]) == 'baldwin':
            y_train[i] = 0
        elif lastname(training_files[i]) == 'carell':
            y_train[i] = 1
           
    # creating labels for validation set      
    y_valid = np.empty([len(validation_files)])
    for i in range(len(validation_files)):
        if lastname(validation_files[i]) == 'baldwin':
            y_valid[i] = 0
        elif lastname(validation_files[i]) == 'carell':
            y_valid[i] = 1
       
    # creating labels for test set
    y_test = np.empty([len(test_files)])
    for i in range(len(test_files)):
        if lastname(test_files[i]) == 'baldwin':
            y_test[i] = 0
        elif lastname(test_files[i]) == 'carell':
            y_test[i] = 1
                  
    X_train = imread('data/' +training_files[0])[:,:,0].flatten()
    for i in range(1, len(training_files)):
        im = imread('data/' + training_files[i])
        X_train = np.vstack((X_train, im[:,:,0].flatten()))
    X_train = X_train/255.
    X_train = X_train.T
    
    X_valid = imread('data/' +validation_files[0])[:,:,0].flatten()
    for i in range(1, len(validation_files)):
        im = imread('data/' + validation_files[i])
        X_valid = np.vstack((X_valid, im[:,:,0].flatten()))
    X_valid = X_valid/255.
    X_valid = X_valid.T
    
    X_test = imread('data/' +test_files[0])[:,:,0].flatten()
    for i in range(1, len(test_files)):
        im = imread('data/' + test_files[i])
        X_test = np.vstack((X_test, im[:,:,0].flatten()))
    X_test = X_test/255.
    X_test = X_test.T
    

    # gradient descent
    prev_t = init_t-10*EPS
    t = init_t.copy()
    
    iter  = 0
    train_accuracies = list()
    valid_accuracies = list()
    epoch = list()
    
    while np.linalg.norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(X_train, y_train, t)
        if iter % freq == 0:
            print "Iter", iter
            print "theta = (%.2f, %.2f, ..., %.2f)" % (t[0], t[1], t[-1]) 
            print "f(theta) = %.2f\n" %(f(X_train, y_train, t))
            print "Gradient: ", df(X_train, y_train, t), "\n"

            Xt = np.vstack((np.ones((1, X_train.shape[1])), X_train))
            train_acc= accuracy(Xt, y_train, t)
            train_accuracies.append(train_acc)

            Xv = np.vstack((np.ones((1, X_valid.shape[1])), X_valid))
            valid_acc= accuracy(Xv, y_valid, t)
            valid_accuracies.append(valid_acc)

            print '\ntraining accuracy: %.2f%%\nvalidation accuracy: %.2f%%\n'% (train_acc*100, valid_acc*100)
            
            epoch.append(iter)
        iter += 1

    # check gradient
    print '--------check gradient--------'
    check_gradient(X_train, y_train, t, 2, h=1e-6)
    check_gradient(X_train, y_train, t, 6, h=1e-6)
    
    # report final performance
    print '--------final performance--------'
    print 'value of quadratic cost function on training set is %.2f'% (f(X_train, y_train, t))
    print 'value of quadratic cost funciton on validation set is %.2f\n'% (f(X_train, y_train, t))
    print 'training accuracy: %.2f%%\nvalidation accuracy: %.2f%%\n'% (train_acc*100, valid_acc*100)
    
    
    # graph performance during training
    plt.plot(epoch, train_accuracies, color='blue')
    plt.plot(epoch, valid_accuracies, color='green')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()
    plt.figure()
    
    return t
   
# bad initialization
#np.random.seed(SEED)
#init_t = np.random.normal(0, 1, size=32*32+1)
#theta = train(training_files, validation_files, test_files, init_t, alpha = 1e-6)

np.random.seed(SEED)
init_t = np.random.normal(0, 0.1, size=32*32+1)
theta = train(training_files, validation_files, test_files, init_t, alpha = 1e-6)

##########################  PART 4  ##########################

# 4 a)
# visualize theta with full training set
im_theta = theta[1:].reshape(32,32)
plt.imshow(im_theta, cmap = plt.cm.coolwarm)
plt.figure()

# train on 2 images for each actor
baldwin_training2 = baldwin[0:2]
carell_training2 = carell[0:2]

training_files2 = baldwin_training2 + carell_training2
np.random.seed(SEED)
np.random.shuffle(training_files2)
theta2 = train(training_files2, validation_files, test_files, init_t, alpha = 5e-6, freq=50)

# visualize theta with 2-image training set
im_theta2 = theta2[1:].reshape(32,32)
plt.imshow(im_theta2, cmap = plt.cm.coolwarm)
plt.figure()

# 4 b)
# visualize theta with faces #

# starting with baldwin #
init_face_b = np.hstack((0, imread('data/baldwin47.jpg')[:,:,0].flatten()/255.))
plt.imshow(init_face_b[1:].reshape(32,32), cmap = plt.cm.gray) 
plt.figure()

# stop early
theta_face_b1 = train(training_files, validation_files, test_files, init_face_b, alpha = 1e-6, max_iter = 3000)
im_b1 = theta_face_b1[1:].reshape(32,32)
plt.imshow(im_b1, cmap = plt.cm.coolwarm) 
plt.figure()

# stop later
theta_face_b2 = train(training_files, validation_files, test_files, init_face_b, alpha = 1e-6, max_iter = 300000)
im_b2 = theta_face_b1[1:].reshape(32,32)
plt.imshow(im_b2, cmap = plt.cm.coolwarm) 
plt.figure()


# starting with carell #
init_face_c = np.hstack((0, imread('data/carell35.jpg')[:,:,0].flatten()/255.))
plt.imshow(init_face_c[1:].reshape(32,32), cmap = plt.cm.gray) 
plt.figure()

# stop early
theta_face_c1 = train(training_files, validation_files, test_files, init_face_c, alpha = 1e-6, max_iter = 10000)
im_c1 = theta_face_c1[1:].reshape(32,32)
plt.imshow(im_c1, cmap = plt.cm.coolwarm) 
plt.figure()

# stop later
theta_face_c2 = train(training_files, validation_files, test_files, init_face_c, alpha = 1e-6, max_iter = 300000)
im_c2 = theta_face_c2[1:].reshape(32,32)
plt.imshow(im_c2, cmap = plt.cm.coolwarm)
plt.figure()


##########################  PART 5  ##########################

# download and cleaning images

other_act=list()
for line in open("faces_other.txt").readlines():
    other_act.append(line.split('\n')[0])
other_actresses = other_act[:3]
other_actors = other_act[3:]

#
#for a in other_actresses:
#    name = a.split()[1].lower()
#    i = 0
#    for line in open("facescrub_actresses.txt"):
#        filetype = line.split()[4].split('.')[-1]
#        if a in line:
#            filename_raw = (name+str(i)+'_raw'+'.'+filetype).lower()
#            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename_raw), {}, 35)
#            if not os.path.isfile("uncropped/"+filename_raw):
#                continue
#            try:
#                im = imread("uncropped/"+filename_raw)
#                filename =name+str(i)+'.'+filetype
#                if len(im.shape) == 3: 
#                    im = rgb2gray(im)
#                x1 = int(line.split()[5].split(',')[0])
#                y1 = int(line.split()[5].split(',')[1])
#                x2 = int(line.split()[5].split(',')[2])
#                y2 = int(line.split()[5].split(',')[3])
#                im = imresize(im[y1:y2, x1:x2], (32, 32))
#                plt.imsave("data/"+filename.split('_')[0], im, cmap = plt.cm.gray)
#            except IOError:
#                pass
#            print filename
#            i += 1
#
#for a in other_actors:
#    name = a.split()[1].lower()
#    i = 0
#    for line in open("facescrub_actors.txt"):
#        filetype = line.split()[4].split('.')[-1]
#        if a in line:
#            filename_raw = (name+str(i)+'_raw'+'.'+filetype).lower()
#            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename_raw), {}, 35)
#            if not os.path.isfile("uncropped/"+filename_raw):
#                continue
#            try:
#                im = imread("uncropped/"+filename_raw)
#                filename =name+str(i)+'.'+filetype
#                if len(im.shape) == 3: 
#                    im = rgb2gray(im)
#                x1 = int(line.split()[5].split(',')[0])
#                y1 = int(line.split()[5].split(',')[1])
#                x2 = int(line.split()[5].split(',')[2])
#                y2 = int(line.split()[5].split(',')[3])
#                im = imresize(im[y1:y2, x1:x2], (32, 32))
#                plt.imsave("data/"+filename.split('_')[0], im, cmap = plt.cm.gray)
#            except IOError:
#                pass
#            print filename
#            i += 1
#
## remove expired and mis-cropped images
#for line in open("badimages.txt"):
#    try:
#        os.remove('data/'+line.split('\n')[0])
#    except OSError:
#        pass

# get file names for different actors
male = list()
female = list()

actors_lastname = [name.split(' ')[1].lower() for name in actors]
actresses_lastname = [name.split(' ')[1].lower() for name in actresses]
other_actors_lastname = [name.split(' ')[1].lower() for name in other_actors]
other_actresses_lastname = [name.split(' ')[1].lower() for name in other_actresses]

for files in os.walk("data/"):  
    files = files[-1]
for filename in files:
    name = lastname(filename)
    if name in other_actors_lastname:
        male.append(filename)
    elif name in other_actresses_lastname:
        female.append(filename)
    

# constract validation data and labels
male.sort()
np.random.seed(SEED)
np.random.shuffle(male)

female.sort()
np.random.seed(SEED)
np.random.shuffle(female)

gender_validation_files = male + female
np.random.seed(SEED)
np.random.shuffle(gender_validation_files)

y_valid = np.empty([len(gender_validation_files)])
for i in range(len(gender_validation_files)):
    if lastname(gender_validation_files[i]) in other_actors_lastname:
        y_valid[i] = 0
    elif lastname(gender_validation_files[i]) in other_actresses_lastname:
        y_valid[i] = 1

X_valid = imread('data/' + gender_validation_files[0])[:,:,0].flatten()
for i in range(1, len(gender_validation_files)):
    im = imread('data/' + gender_validation_files[i])
    X_valid = np.vstack((X_valid, im[:,:,0].flatten()))
X_valid = X_valid/255.
X_valid = X_valid.T

def train_gender(training_files, X_valid, y_valid, init_t, alpha = 1e-6, EPS=1e-5, freq=500, max_iter=300000):
    M = len(training_files)
    N = 32*32
    
    # creating labels for training set  
    y_train = np.empty([len(training_files)])
    for i in range(len(training_files)):
        if lastname(training_files[i]) in actors_lastname:
            y_train[i] = 0
        elif lastname(training_files[i]) in actresses_lastname:
            y_train[i] = 1
    
    X_train = imread('data/' +training_files[0])[:,:,0].flatten()
    for i in range(1, len(training_files)):
        im = imread('data/' + training_files[i])
        X_train = np.vstack((X_train, im[:,:,0].flatten()))
    X_train = X_train/255.
    X_train = X_train.T

    
    # gradient descent
    prev_t = init_t-10*EPS
    t = init_t.copy()
    
    iter  = 0
    count = 0
    valid_acc = 0
    while np.linalg.norm(t - prev_t) > EPS and count < 5 and iter < max_iter:
        prev_t = t.copy()
        prev_valid_acc = valid_acc
        t -= alpha*df(X_train, y_train, t)
        if iter % freq == 0:
            print "Iter", iter
            print "theta = (%.2f, %.2f, ..., %.2f), f(theta) = %.2f" % (t[0], t[1], t[-1], f(X_train, y_train, t)) 
            print "Gradient: ", df(X_train, y_train, t), "\n"
            Xt = np.vstack((np.ones((1, X_train.shape[1])), X_train))
            train_acc= accuracy(Xt, y_train, t)

            Xv = np.vstack((np.ones((1, X_valid.shape[1])), X_valid))
            valid_acc= accuracy(Xv, y_valid, t)

            print '\ntraining accuracy: %.2f%%\nvalidation accuracy: %.2f%%\n'% (train_acc*100, valid_acc*100)
            if train_acc <= 0.85:
                delta = 1
            else:
                delta = valid_acc - prev_valid_acc
                if delta < 0:
                    count += 1
        iter += 1

    # report final performance
    print '--------final performance--------'
    print 'training size = %1d' % (M)
    print '\ntraining accuracy: %.2f%%\nvalidation accuracy: %.2f%%\n'% (train_acc*100, valid_acc*100) 

    return train_acc, valid_acc

# training for various training set sizes
train_accuracies = list()
valid_accuracies = list()
size = list()
for i in [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    training_gender = bracco[0:i] + gilpin[0:i] + harmon[0:i] \
                        + baldwin[0:i] + hader[0:i] + carell[0:i]
    np.random.seed(SEED)
    np.random.shuffle(training_gender)
    if (i <= 20):
        a = 1e-6
    else:
        a = 5e-6
    train_acc, valid_acc = train_gender(training_gender, X_valid, y_valid, init_t, alpha = a)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)
    size.append(i)

# plot performance vs. training set size
plt.plot(size, train_accuracies, color='blue')
plt.plot(size, valid_accuracies, color='green')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('training set size')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()
plt.figure()


##########################  PART 6  ##########################

# 6 c)
def J(x, y, theta):
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return np.sum((np.dot(theta.T, x)-y) ** 2)
    

def dJ(x, y, theta):
    x = np.vstack((np.ones((1, x.shape[1])), x))
    return 2*np.dot(x, (np.dot(theta.T, x)-y).T)

# 6 d)
def check_gradient_multi(theta, x, y, i, j, h=1e-7):
    deltaij = np.zeros(theta.shape)
    deltaij[i, j] = h
    fd = (J(x, y, theta + deltaij) - J(x, y, theta))/h
    dj = dJ(x, y, theta)[i, j]
    print 'theta [%1d, %1d]'%(i, j)
    print '--------------------------------------'
    print 'finite-difference approximation: %.7f\ndJ/dthetaij: %.7f' % (fd, dj)
    print 'absolute difference: %.7f' % (abs(fd - dj))

# construct training, validation and test sets for part 7
training_files_multi = bracco[0:70] + gilpin[0:70] + harmon[0:70] \
                        + baldwin[0:70] + hader[0:70] + carell[0:70]
np.random.seed(SEED)
np.random.shuffle(training_files_multi)

validation_files_multi =  bracco[70:80] + gilpin[70:80] + harmon[70:80] \
                        + baldwin[70:80] + hader[70:80] + carell[70:80]
np.random.seed(SEED)
np.random.shuffle(validation_files_multi)

test_files_multi = bracco[80:90] + gilpin[80:90] + harmon[80:90] \
                        + baldwin[80:90] + hader[80:90] + carell[80:90]
np.random.seed(SEED)
np.random.shuffle(test_files_multi)


# contract labels and data for calculation
y_train = np.empty([len(training_files_multi), 6])
for i in range(len(training_files_multi)):
    z = np.zeros(6)
    if lastname(training_files_multi[i]) == 'bracco':
        z[0] = 1
        y_train[i] = z
    elif lastname(training_files_multi[i]) == 'gilpin':
        z[1] = 1
        y_train[i] = z        
    elif lastname(training_files_multi[i]) == 'harmon':
        z[2] = 1
        y_train[i] = z        
    elif lastname(training_files_multi[i]) == 'baldwin':
        z[3] = 1
        y_train[i] = z        
    elif lastname(training_files_multi[i]) == 'hader':
        z[4] = 1
        y_train[i] = z        
    elif lastname(training_files_multi[i]) == 'carell':
        z[5] = 1
        y_train[i] = z 
y_train = y_train.T

X_train = imread('data/' +training_files_multi[0])[:,:,0].flatten()
for i in range(1, len(training_files_multi)):
    im = imread('data/' + training_files_multi[i])
    X_train = np.vstack((X_train, im[:,:,0].flatten()))
X_train = X_train/255.
X_train = X_train.T

# check gradient at different components of theta
np.random.seed(SEED)
init_t_multi = np.random.normal(0, 0.1, size=(32*32+1, 6))
check_gradient_multi(init_t_multi, X_train, y_train, 2, 4, h=1e-7)
check_gradient_multi(init_t_multi, X_train, y_train, 456, 2, h=1e-7)
check_gradient_multi(init_t_multi, X_train, y_train, 304, 0, h=1e-7)
check_gradient_multi(init_t_multi, X_train, y_train, 100, 5, h=1e-7)
check_gradient_multi(init_t_multi, X_train, y_train, 230, 3, h=1e-7)



##########################  PART 7  ##########################

def multi_accuracy(x, y, t):
    M = float(y.shape[1])
    tTx = np.dot(t.T, x)
    count = 0
    for i in range(y.shape[1]):
        if np.argmax(tTx[:, i]) == np.argmax(y[:, i]):
            count += 1
    return count/M
        

def train_multi(training_files, validation_files, test_files, init_t, alpha = 1e-6, EPS=1e-5, freq=500, max_iter=300000):
    M = len(training_files)
    N = 32*32
    
    # creating labels for training set  
    y_train = np.empty([len(training_files), 6])
    for i in range(len(training_files)):
        z = np.zeros(6)
        if lastname(training_files[i]) == 'bracco':
            z[0] = 1
            y_train[i] = z
        elif lastname(training_files[i]) == 'gilpin':
            z[1] = 1
            y_train[i] = z        
        elif lastname(training_files[i]) == 'harmon':
            z[2] = 1
            y_train[i] = z        
        elif lastname(training_files[i]) == 'baldwin':
            z[3] = 1
            y_train[i] = z        
        elif lastname(training_files[i]) == 'hader':
            z[4] = 1
            y_train[i] = z        
        elif lastname(training_files[i]) == 'carell':
            z[5] = 1
            y_train[i] = z 
    y_train = y_train.T
            
    # creating labels for validation set  
    y_valid = np.empty([len(validation_files), 6])
    for i in range(len(validation_files)):
        z = np.zeros(6)
        if lastname(validation_files[i]) == 'bracco':
            z[0] = 1
            y_valid[i] = z
        elif lastname(validation_files[i]) == 'gilpin':
            z[1] = 1
            y_valid[i] = z        
        elif lastname(validation_files[i]) == 'harmon':
            z[2] = 1
            y_valid[i] = z        
        elif lastname(validation_files[i]) == 'baldwin':
            z[3] = 1
            y_valid[i] = z        
        elif lastname(validation_files[i]) == 'hader':
            z[4] = 1
            y_valid[i] = z        
        elif lastname(validation_files[i]) == 'carell':
            z[5] = 1
            y_valid[i] = z
    y_valid = y_valid.T
    
    # creating labels for test set  
    y_test = np.empty([len(test_files), 6])
    for i in range(len(test_files)):
        z = np.zeros(6)
        if lastname(test_files[i]) == 'bracco':
            z[0] = 1
            y_test[i] = z
        elif lastname(test_files[i]) == 'gilpin':
            z[1] = 1
            y_test[i] = z        
        elif lastname(test_files[i]) == 'harmon':
            z[2] = 1
            y_test[i] = z        
        elif lastname(test_files[i]) == 'baldwin':
            z[3] = 1
            y_test[i] = z        
        elif lastname(test_files[i]) == 'hader':
            z[4] = 1
            y_test[i] = z        
        elif lastname(test_files[i]) == 'carell':
            z[5] = 1
            y_test[i] = z
    y_test = y_test.T
       
    
    X_train = imread('data/' +training_files[0])[:,:,0].flatten()
    for i in range(1, len(training_files)):
        im = imread('data/' + training_files[i])
        X_train = np.vstack((X_train, im[:,:,0].flatten()))
    X_train = X_train/255.
    X_train = X_train.T
    
    X_valid = imread('data/' +validation_files[0])[:,:,0].flatten()
    for i in range(1, len(validation_files)):
        im = imread('data/' + validation_files[i])
        X_valid = np.vstack((X_valid, im[:,:,0].flatten()))
    X_valid = X_valid/255.
    X_valid = X_valid.T
    
    X_test = imread('data/' +test_files[0])[:,:,0].flatten()
    for i in range(1, len(test_files)):
        im = imread('data/' + test_files[i])
        X_test = np.vstack((X_test, im[:,:,0].flatten()))
    X_test = X_test/255.
    X_test = X_test.T

    # gradient descent
    np.random.seed(SEED)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    
    iter  = 0
    train_accuracies = list()
    valid_accuracies = list()
    epoch = list()
    while np.linalg.norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*dJ(X_train, y_train, t)
        if iter % freq == 0:
            print "Iter", iter
            print "J(theta) = %.2f" % (J(X_train, y_train, t)) 
            print "\n"
            Xt = np.vstack((np.ones((1, X_train.shape[1])), X_train))
            train_acc= multi_accuracy(Xt, y_train, t)
            train_accuracies.append(train_acc)
            
            Xv = np.vstack((np.ones((1, X_valid.shape[1])), X_valid))
            valid_acc= multi_accuracy(Xv, y_valid, t)
            valid_accuracies.append(valid_acc)
            print 'training accuracy: %.2f%%\nvalidation accuracy: %.2f%%\n'% (train_acc*100, valid_acc*100)
            epoch.append(iter)
        iter += 1


    # report final performance
    print '--------final performance--------'
    Xt = np.vstack((np.ones((1, X_train.shape[1])), X_train))
    train_acc= multi_accuracy(Xt, y_train, t)
                
    Xv = np.vstack((np.ones((1, X_valid.shape[1])), X_valid))
    valid_acc= multi_accuracy(Xv, y_valid, t)
    print 'training accuracy: %.2f%%\nvalidation accuracy: %.2f%%\n'% (train_acc*100, valid_acc*100)
    
    plt.plot(epoch, train_accuracies, color='blue')
    plt.plot(epoch, valid_accuracies, color='green')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()
    plt.figure()
    
    return t


np.random.seed(SEED)
init_t_multi = np.random.normal(0, 0.1, size=(32*32+1, 6))
theta_multi = train_multi(training_files_multi, validation_files_multi, test_files_multi, init_t_multi, alpha = 5e-6, EPS=1e-5, max_iter=500000)



##########################  PART 8  ##########################

print 'bracco at index %d' % (np.argmax(np.dot(theta_multi.T, np.hstack((0, imread('data/bracco107.jpg')[:,:,0].flatten()/255.)))))
print 'gilpin at index %d' % (np.argmax(np.dot(theta_multi.T, np.hstack((0, imread('data/gilpin44.jpg')[:,:,0].flatten()/255.)))))
print 'harmon at index %d' % (np.argmax(np.dot(theta_multi.T, np.hstack((0, imread('data/harmon101.jpg')[:,:,0].flatten()/255.)))))
print 'baldwin at index %d' % (np.argmax(np.dot(theta_multi.T, np.hstack((0, imread('data/baldwin109.jpg')[:,:,0].flatten()/255.)))))
print 'hader at index %d' % (np.argmax(np.dot(theta_multi.T, np.hstack((0, imread('data/hader37.jpg')[:,:,0].flatten()/255.)))))
print 'carell at index %d' % (np.argmax(np.dot(theta_multi.T, np.hstack((0, imread('data/carell115.jpg')[:,:,0].flatten()/255.)))))

# bracco
im_face_multi0 = theta_multi[:, 0][1:].reshape(32, 32)
plt.imshow(im_face_multi0, cmap = plt.cm.coolwarm) 
plt.figure()
# gilpin
im_face_multi1 = theta_multi[:, 1][1:].reshape(32, 32)
plt.imshow(im_face_multi1, cmap = plt.cm.coolwarm) 
plt.figure()
# harmon
im_face_multi2 = theta_multi[:, 2][1:].reshape(32, 32)
plt.imshow(im_face_multi2, cmap = plt.cm.coolwarm) 
plt.figure()
# baldwin
im_face_multi3 = theta_multi[:, 3][1:].reshape(32, 32)
plt.imshow(im_face_multi3, cmap = plt.cm.coolwarm) 
plt.figure()
# hader
im_face_multi4 = theta_multi[:, 4][1:].reshape(32, 32)
plt.imshow(im_face_multi4, cmap = plt.cm.coolwarm) 
plt.figure()
# carell
im_face_multi5 = theta_multi[:, 5][1:].reshape(32, 32)
plt.imshow(im_face_multi5, cmap = plt.cm.coolwarm) 
plt.figure()
