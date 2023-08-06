import pycbc.noise
import pycbc.psd
import matplotlib.pyplot as plt
import numpy as np


def create_noise():
    # The color of the noise matches a PSD which you provide
    flow = 30.0
    delta_f = 1.0 / 32
    flen = int(4096 / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    # Generate 32 seconds of noise at 4096 Hz
    delta_t = 1.0 / 8192
    tsamples = int(32 / delta_t)
    ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd)
    return ts, psd


if __name__ == "__main__":
    ts, psd = create_noise()
    print(len(ts.sample_times)*ts.delta_t)
    plt.plot(ts.sample_times, ts)
    plt.ylabel('Strain')
    plt.xlabel('Time (s)')
    plt.show()
