# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:06:49 2019

@author: Anthony
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'spectro_gen'


# Spectrogram generation
height = 500
width = 500
N = 100
N_test = 100

std_noise = 0.1

min_lines = 1
max_lines = 8
width_line = 21

noise = np.abs(np.random.randn(height, width, N) * std_noise)
noise_test = np.abs(np.random.randn(height, width, N_test) * std_noise)

imgs = np.zeros((N, height, height))
imgs_test = np.zeros((N_test, height, height))

x = np.linspace(-1, 1, width_line)
y = np.exp(-x**2)/0.747
y = y[:, None]

for i in range(N):
    lines = np.random.randint(min_lines, max_lines+1)
    for line in range(lines):
        idx = np.random.randint(0, height-width_line+1)
        scale = np.abs(np.random.randn())
        noise[idx:idx+width_line, :, i] += y * scale

for i in range(N_test):
    lines = np.random.randint(min_lines, max_lines+1)
    for line in range(lines):
        idx = np.random.randint(0, height-width_line+1)
        scale = np.abs(np.random.randn())
        noise_test[idx:idx+width_line, :, i] += y * scale

print('Generation completed')

# Resize and normalise spectrograms for CNN
#for i in range(N):
#    imgs[i] = cv2.resize(noise[..., i], (height, height))

#print('Resize completed')
imgs = noise.transpose((2, 0, 1))
imgs_test = noise_test.transpose((2, 0, 1))

mean = imgs.mean(axis=0)
std = imgs.std(axis=0)

imgs = (imgs - mean)/(std + 1e-8)
imgs = imgs[..., None]

imgs_test = (imgs_test - mean)/(std + 1e-8)
imgs_test = imgs_test[..., None]

print('Normalisation completed')


from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Reshape, UpSampling2D, ZeroPadding2D
from keras.layers import Dense, Flatten
from keras.models import Model

# Build network
inputs = Input(shape=(height, height, 1))

"""
x = Conv2D(8, 3, padding='same', activation='relu')(inputs) # 500
x = MaxPooling2D(4)(x)
x = Conv2D(16, 3, padding='same', activation='relu')(x) # 125
x = MaxPooling2D(4)(x)
x = Conv2D(32, 3, padding='same', activation='relu')(x) # 31
x = MaxPooling2D(4)(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x) # 7
x = MaxPooling2D(2)(x) # 3

x = Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dense(3*3*64, activation='relu')(x)

x = Reshape((3, 3, 64))(x) # 3
x = UpSampling2D(2)(x)
x = ZeroPadding2D(((0, 1), (0, 1)))(x) # 7
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = UpSampling2D(4)(x)
x = ZeroPadding2D(((0, 3), (0, 3)))(x) # 31
x = Conv2D(16, 3, padding='same', activation='relu')(x)
x = UpSampling2D(4)(x)
x = ZeroPadding2D(((0, 1), (0, 1)))(x) # 125
x = Conv2D(8, 3, padding='same', activation='relu')(x)
x = UpSampling2D(4)(x) # 500
"""


x = Conv2D(8, 3, padding='same', activation='relu')(inputs) # 500
x = MaxPooling2D(2)(x)
x = Conv2D(8, 3, padding='same', activation='relu')(x) # 250
x = MaxPooling2D(2)(x)
x = Conv2D(16, 3, padding='same', activation='relu')(x) # 125
x = MaxPooling2D(2)(x)
x = Conv2D(16, 3, padding='same', activation='relu')(x) # 62
x = MaxPooling2D(2)(x)
x = Conv2D(32, 3, padding='same', activation='relu')(x) # 31
x = MaxPooling2D(2)(x)
x = Conv2D(32, 3, padding='same', activation='relu')(x) # 15
x = MaxPooling2D(2)(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x) # 7
x = MaxPooling2D(2)(x) # 3

x = Flatten()(x)
x = Dense(50, activation='relu')(x)
x = Dense(3*3*64, activation='relu')(x)

x = Reshape((3, 3, 64))(x) # 3
x = UpSampling2D(2)(x)
x = ZeroPadding2D(((0, 1), (0, 1)))(x) # 7
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = UpSampling2D(2)(x)
x = ZeroPadding2D(((0, 1), (0, 1)))(x) # 15
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = UpSampling2D(2)(x)
x = ZeroPadding2D(((0, 1), (0, 1)))(x) # 31
x = Conv2D(16, 3, padding='same', activation='relu')(x)
x = UpSampling2D(2)(x) # 62
x = Conv2D(16, 3, padding='same', activation='relu')(x)
x = UpSampling2D(2)(x)
x = ZeroPadding2D(((0, 1), (0, 1)))(x) # 125
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = UpSampling2D(2)(x) # 250
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = UpSampling2D(2)(x) # 500


outputs = Conv2D(1, 3, padding='same', activation='relu')(x)


# Build model
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
hist = model.fit(imgs, imgs, batch_size=50, epochs=10, validation_data=(imgs_test, imgs_test))

model.summary()




sample = imgs[None, 20]
pred = model.predict(sample)

sample = sample.squeeze()
pred = pred.squeeze()

sample = sample * (std + 1e-8) + mean
pred = pred * (std + 1e-8) + mean

plt.figure('original')
plt.imshow(sample)
plt.figure('predict')
plt.imshow(pred)

print('mean square loss: ', np.square(pred - sample).mean())

plt.figure('loss')
plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])