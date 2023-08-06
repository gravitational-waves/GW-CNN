# *****************************************************************************

# KIDE IN /home/ccs/miniconda3/envs/gw_detection/lib/python3.6/site-packages/pycbc/filter/resample.py

# *****************************************************************************

from pycbc.frame import read_frame
from pycbc import catalog
from pycbc.filter import resample_to_delta_t, highpass, sigma
from pycbc.filter.matchedfilter import matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform
import matplotlib.pyplot as plt


# print(timeseries.sample_rate)
# plt.plot(ts.sample_times, ts)
# plt.show()


def matched_f(ts):
    ts = resample_to_delta_t(highpass(ts, 15.0), 1.0/2048)

    ts = ts.crop(2, 2)
    # print(conditioned.delta_t)
    psd = ts.psd(4)
    psd = interpolate(psd, ts.delta_f)
    psd = inverse_spectrum_truncation(psd, 4 * ts.sample_rate, low_frequency_cutoff=15)

    m = 36
    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                             mass1=m,
                             mass2=m,
                             delta_t=ts.delta_t,
                             f_lower=20)
    hp.resize(len(ts))
    template = hp.cyclic_time_shift(hp.start_time)

    snr = matched_filter(template, ts, psd=psd, low_frequency_cutoff=20)
    snr = snr.crop(4+4, 4)

    peak = abs(snr).numpy().argmax()
    snrp = snr[peak]
    merger_time = snr.sample_times[peak]

    print("We found a signal at {}s with SNR {}".format(merger_time, abs(snrp)))

    # Shift the template to the peak time
    dt = merger_time - ts.start_time
    aligned = template.cyclic_time_shift(dt)

    # Scale the template so that it would have SNR 1 in this data
    aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)

    # Scale the template amplitude and phase to the peak value
    aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
    aligned.start_time = ts.start_time

    # We do it this way so that we can whiten both the template and the data
    white_data = (ts.to_frequencyseries() / psd**0.5).to_timeseries()

    # apply a smoothing of the turnon of the template to avoid a transient
    # from the sharp turn on in the waveform.
    tapered = aligned.highpass_fir(30, 512, remove_corrupted=False)
    white_template = (tapered.to_frequencyseries() / psd**0.5).to_timeseries()

    white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
    white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)

    # Select the time around the merger
    white_data = white_data.time_slice(merger_time-.5, merger_time+.5)
    white_template = white_template.time_slice(merger_time-.5, merger_time+.5)

    plt.figure(figsize=[15, 5])
    plt.plot(white_data.sample_times, white_data, label="Data")
    plt.plot(white_template.sample_times, white_template, label="Template")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_name = "data/H-H1_LOSC_4_V2-1126259446-32.gwf"
    # file_name = "data/H-H1_LOSC_4_V2-1126257414-4096.gwf"
    channel_name = "H1:LOSC-STRAIN"
    start = 1126259446
    end = start + 32

    merger = catalog.Merger("GW150914")
    # merger.time is 1126259462.4

    timeseries = read_frame(file_name, channel_name)

    matched_f(timeseries)
