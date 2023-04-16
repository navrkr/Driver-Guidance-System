import scipy.misc
import cv2
import random
import numpy as np

xs = []
ys = []
accels = []
brake = []
# gear = []
opticalFlow = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0
corr_train_batch_pointer = 0
corr_val_batch_pointer = 0

dataPath = "indian_dataset/"
corrDataPath = "indian_dataset/corr/"
fileNamePrefix = "circuit2_x264.mp4 "
#read data.txt
with open(dataPath+"data.txt") as f:
# with open("driving_dataset/data.txt") as f:
    for line in f:
        # xs.append("driving_dataset/" + line.split()[0])
        xs.append(dataPath + fileNamePrefix + str(int(line.split()[0])).zfill(5)+".jpg")
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)


with open(corrDataPath+"optFlow.txt") as f:
# with open("driving_dataset/data.txt") as f:
    for line in f:
        # xs.append("driving_dataset/" + line.split()[0])
        opticalFlow.append(float(line.split()[0]) * scipy.pi / 180)


#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], (200, 66)) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], (200, 66)) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out