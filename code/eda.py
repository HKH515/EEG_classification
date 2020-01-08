import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import collections
import librosa
import sys

path = sys.argv[1]

data = np.load(path, allow_pickle=True)

x = data['x']
y = data['y']

fig_1 = plt.figure(figsize=(12, 6))
plt.plot(x[100, ...].ravel())
plt.title("EEG Epoch")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.show()

fig_2 = plt.figure(figsize=(12, 6))
plt.plot(y.ravel())
plt.title("Sleep Stages")
plt.ylabel("Classes")
plt.xlabel("Time")
plt.show()