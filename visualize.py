# from matched_filtering import matched_f
# from matplotlib import pyplot as plt
# from pycbc.waveform import TimeSeries
import numpy as np
import pandas as pd

training_data = np.load("data/training_data.dat")
training_data_noise = np.load("data/training_data_noise.dat")
testing_data = np.load("data/testing_data.dat")
testing_data_noise = np.load("data/testing_data_noise.dat")

training_features = []
training_labels = []
testing_features = []
testing_labels = []

for i in training_data:
    training_labels.append(i[1])
    training_features.append(i[0])
for i in training_data_noise:
    training_labels.append(i[1])
    training_features.append(i[0])
for i in testing_data:
    testing_labels.append(i[1])
    testing_features.append(i[0])
for i in testing_data_noise:
    testing_labels.append(i[1])
    testing_features.append(i[0])

training_features = np.array(training_features)
training_labels = np.array(training_labels)
testing_features = np.array(testing_features)
testing_labels = np.array(testing_labels)

training_features_series = pd.DataFrame(training_features)
print(training_features_series.describe())
