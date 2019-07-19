# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:49:04 2019

@author: Anthony
"""

import numpy as np
import matplotlib.pyplot as plt

path = 'spectro_gen'


# Spectrogram generation
height = 30
width = 30
N = 10000
N_test = 1000

std_noise = 0.1

min_lines = 1
max_lines = 5
width_line = 3

imgs = np.abs(np.random.randn(N, height, width) * std_noise)
imgs_test = np.abs(np.random.randn(N_test, height, width) * std_noise)

x = np.linspace(-1, 1, width_line)
y = np.exp(-x**2)/0.747
y = y[:, None]

for i in range(N):
    lines = np.random.randint(min_lines, max_lines+1)
    for line in range(lines):
        idx = np.random.randint(0, height-width_line+1)
        scale = np.abs(np.random.randn())
        imgs[i, idx:idx+width_line, :] += y * scale
        
for i in range(N_test):
    lines = np.random.randint(min_lines, max_lines+1)
    for line in range(lines):
        idx = np.random.randint(0, height-width_line+1)
        scale = np.abs(np.random.randn())
        imgs_test[i, idx:idx+width_line, :] += y * scale

print('Generation completed')


# Normalise for training
mean = np.mean(imgs, axis=0)
std = np.std(imgs, axis=0)

imgs = (imgs - mean)/(std + 1e-8)
imgs = imgs[..., None]

imgs_test = (imgs_test - mean)/(std + 1e-8)
imgs_test = imgs_test[..., None]






from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Reshape, UpSampling2D, ZeroPadding2D
from keras.layers import Dense, Flatten
from keras.models import Model

inputs = Input(shape=(height, width, 1))

x = Conv2D(16, 3, padding='same', activation='relu')(inputs)
x = MaxPooling2D(2)(x)
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = MaxPooling2D(2)(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = MaxPooling2D(2)(x)

x = Flatten()(x)
x = Dense(20, activation='relu')(x)
x = Dense(3*3*64, activation='relu')(x)
x = Reshape((3, 3, 64))(x)

x = UpSampling2D(2)(x)
x = ZeroPadding2D(((0, 1), (0, 1)))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = UpSampling2D(2)(x)
x = ZeroPadding2D(((0, 1), (0, 1)))(x)
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = UpSampling2D(2)(x)
x = Conv2D(16, 3, padding='same', activation='relu')(x)

outputs = Conv2D(1, 3, padding='same', activation='relu')(x)


model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
hist = model.fit(imgs, imgs, batch_size=40, epochs=20, validation_data=(imgs_test, imgs_test))

model.summary()




#sample = plt.imread('spectro gen.png')[..., 0]
#sample = (sample - mean)/(std + 1e-8)
#sample = sample[None, ..., None]
sample = imgs_test[None, 157]
pred = model.predict(sample)

sample = sample.reshape((height, width))
pred = pred.reshape((height, width))

sample = sample * (std + 1e-8) + mean
pred = pred * (std + 1e-8) + mean


plt.figure('original')
plt.imshow(sample)
plt.figure('predict')
plt.imshow(pred)

print('mean square loss: ', np.square(pred - sample).mean())

plt.figure('loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
