from pycbc_noise import create_noise
from pycbc.waveform import get_td_waveform
import numpy as np
from matched_filtering import matched_f
from matplotlib import pyplot as plt
import pandas as pd

multiplier = 1e+20
# 22 low
# 20 medium
# 18 high
# 16 vhigh
# 14 vvhigh
nm = 10000
# 100 low
# 10000 medium
# 1000000 high
# 100000000 vhigh
# 10000000000 vvhigh


def create_templates(m1, m2, noise_multiplier=nm):
    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                             mass1=m1,
                             mass2=m2,
                             delta_t=1.0 / 8192,
                             f_lower=20)

    noise, noise_psd = create_noise()
    noise *= noise_multiplier

    mid = int(len(noise) / 2)
    for i in range(len(hp)):
        noise[mid - i] += hp[len(hp) - i - 1]
    disp = np.random.uniform(-0.1, 0.3)
    disp = int(disp*8192)
    windowed_noise = noise[mid-disp-8192:mid-disp]
    # print(len(windowed_noise))
    # print(windowed_noise.start_time)
    # plt.plot(hp.sample_times, hp)
    # plt.show()
    # plt.plot(noise.sample_times, noise)
    # plt.ylabel('Strain')
    # plt.xlabel('Time (s)')
    # plt.show()
    # matched_f(noise)
    return windowed_noise.numpy()


if __name__ == "__main__":
    choices = [1, 2, 3, 4]

    if 1 in choices:     # Training - data
        data = []
        print("\nCreating data for training")
        for mass1 in np.arange(10, 60, 1):
            for mass2 in np.arange(mass1, 60, 1):
                raw_data = create_templates(mass1, mass2)
                raw_data *= multiplier
                # raw_data = np.ones((4096,))
                data.append([raw_data, 1])
            print("Done for mass:", mass1)
        np.array(data).dump("data/training_data_mn.dat")
        print("Train data creation complete")

    if 2 in choices:     # Training - noise
        data = []
        num = 1275
        print("\nCreating noise for training")
        for i in range(num):
            noise, noise_psd = create_noise()
            noise *= nm
            mid = int(len(noise) / 2)
            noise = noise[mid:mid + 8192]
            noise *= multiplier
            # noise = np.zeros((4096,))
            data.append([noise, 0])
            if i % 100 == 0 or i == num-1:
                print("Iteration: {}".format(i+1))
        np.array(data).dump("data/training_data_noise_mn.dat")
        print("Train noise creation complete")

    if 3 in choices:     # Testing - data
        data = []
        print("\nCreating data for testing")
        for mass1 in np.arange(10.5, 60.5, 1):
            for mass2 in np.arange(mass1, 60.5, 1):
                raw_data = create_templates(mass1, mass2)
                raw_data *= multiplier
                # raw_data = np.ones((4096,))
                data.append([raw_data, 1])
            print("Done for mass:", mass1)
        np.array(data).dump("data/testing_data_mn.dat")
        print("Test data creation complete")

    if 4 in choices:     # Testing - noise
        data = []
        num = 1275
        print("\nCreating noise for testing")
        for i in range(num):
            noise, noise_psd = create_noise()
            noise *= nm
            mid = int(len(noise) / 2)
            noise = noise[mid:mid + 8192]
            noise *= multiplier
            # noise = np.zeros((4096,))
            data.append([noise, 0])
            if i % 100 == 0 or i == num-1:
                print("Iteration: {}".format(i+1))
        np.array(data).dump("data/testing_data_noise_mn.dat")
        print("Test noise creation complete")
