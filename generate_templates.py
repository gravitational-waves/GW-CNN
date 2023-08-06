from pycbc.waveform import get_td_waveform
from matplotlib import pyplot as plt
from pycbc_noise import create_noise
from matched_filtering import matched_f
from pycbc.psd import interpolate, inverse_spectrum_truncation
import pandas as pd


hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                         mass1=10,
                         mass2=10,
                         delta_t=1.0/4096,
                         f_lower=20)

noise, noise_psd = create_noise()
# noise *= 1000

wave_df = pd.DataFrame(hp.numpy())
noise_df = pd.DataFrame(noise.numpy())

print("Wave")
print(wave_df.describe())
print("Noise")
print(noise_df.describe())
# exit()

mid = int(len(noise)/2)
# print(len(hp))

# print(mid)

for i in range(len(hp)):
    noise[mid-i] += hp[len(hp)-i-1]

# plt.plot(noise.sample_times, noise, label="noise")
# plt.plot(hp.sample_times, hp, label="wave")
# plt.grid()
# plt.ylabel("Strain")
# plt.xlabel("Time(s)")
# plt.legend()
# plt.show()

raw_data = noise.copy()
# print(raw_data.delta_f, psd.delta_f)
# psd = raw_data.psd(4)
# psd = interpolate(noise_psd, raw_data.delta_f)
# psd = inverse_spectrum_truncation(psd, 4 * raw_data.sample_rate, low_frequency_cutoff=15)

# white_raw_data = (raw_data.to_frequencyseries() / psd**0.5).to_timeseries()

matched_f(raw_data)
print(raw_data.sample_rate)
print(raw_data.delta_t)
# plt.plot(raw_data)
# plt.show()
