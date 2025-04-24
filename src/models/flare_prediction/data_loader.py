import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime, timezone, timedelta
from astropy.io import fits
from sunpy.visualization.colormaps import color_tables as ct
from matplotlib.dates import *
import matplotlib.image as mpimg
import pandas as pd
import sunpy.map
import sunpy.io
import os

# Establish connection to JSOC

c = drms.Client()

# Define the series
print(c.series(r'hmi\.sharp_'))

si = c.info('hmi.sharp_cea_720s_nrt')

keywords = list(si.keywords.index)
print(keywords)

end_time = datetime.now(timezone.utc)

start_time = end_time - timedelta(hours = 2)

jsoc_time_format = "%Y.%m.%d_%H:%M:%S_TAI"
start_str = start_time.strftime(jsoc_time_format)
end_str = end_time.strftime(jsoc_time_format)

keys = c.query(f'hmi.sharp_cea_720s_nrt[][{start_str} - {end_str}][]', key = keywords)

print(keys)

output_dir = "data/sharp_csv"
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame(keys)
df.to_csv(os.path.join(output_dir, "nrt_sharp"), index = False)

def parse_tai_string(tstr, datetime=True):
    year   = int(tstr[:4])
    month  = int(tstr[5:7])
    day    = int(tstr[8:10])
    hour   = int(tstr[11:13])
    minute = int(tstr[14:16])
    if datetime: 
        return datetime(year,month,day,hour,minute)
    else: 
        return year,month,day,hour,minute