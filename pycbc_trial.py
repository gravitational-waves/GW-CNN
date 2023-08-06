from pycbc.frame import read_frame
from pycbc import catalog
import matplotlib.pyplot as plt
from pprint import pprint

# m = catalog.Merger("GW150914")
# pprint(m.data['median1d'])

# file_name = "data/H-H1_LOSC_4_V2-1126257414-4096.gwf"
# channel_name = "H1:LOSC-STRAIN"
file_name = "data/H-H1_LOSC_16_V1-1126256640-4096.gwf"
channel_name = "H1:GWOSC-16KHZ_R1_STRAIN"
start = 1126257414
end = start + 4096
start_event = 1126259462 - 16
duration = 32

ts = read_frame(file_name, channel_name)
# pprint(ts.__dict__)

# print(ts)
#
# zoom = ts.time_slice(m.time-0.5, m.time+0.5)

plt.plot(ts.sample_times, ts)
plt.show()
