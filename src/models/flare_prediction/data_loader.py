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
from sunpy.net import attrs as a, hek
from sunpy.time import parse_time

def parse_tai_string(tstr, return_datetime=True):
    year   = int(tstr[:4])
    month  = int(tstr[5:7])
    day    = int(tstr[8:10])
    hour   = int(tstr[11:13])
    minute = int(tstr[14:16])
    if return_datetime: 
        return datetime(year,month,day,hour,minute)
    else: 
        return year,month,day,hour,minute

# Establish connection to JSOC

def data_loader(number_of_days = 1):
    c = drms.Client()

    # Define the series
    # print(c.series(r'hmi\.sharp_'))

    si = c.info('hmi.sharp_cea_720s')

    end_time = datetime.now(timezone.utc) - timedelta(days = 100)

    start_time = end_time - timedelta(days = 1)

    # Initialize or load existing DataFrame and CSV
    output_dir = "data/sharp_csv"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "testing_data.csv")

    if os.path.exists(csv_path):
        combined_df = pd.read_csv(csv_path)
    else:
        combined_df = pd.DataFrame()

    for i in range(number_of_days):
        start_time = start_time - timedelta(days=1)
        end_time = end_time - timedelta(days=1)

        jsoc_time_format = "%Y.%m.%d_%H:%M:%S_TAI"
        start_str = start_time.strftime(jsoc_time_format)
        end_str = end_time.strftime(jsoc_time_format)

        # Loading the data
        data = c.query(f'hmi.sharp_cea_720s[][{start_str} - {end_str}@1h][]', 
                    key='T_REC, NOAA_AR, USFLUX,TOTUSJH,TOTPOT,MEANGBT,MEANGBZ,MEANJZD,MEANPOT,MEANSHR,SHRGT45,R_VALUE,AREA_ACR'
        )

        df = pd.DataFrame(data)
        key_cols = ['T_REC', 'NOAA_AR', 'USFLUX', 'TOTUSJH', 'TOTPOT', 'MEANGBT', 'MEANGBZ', 'MEANJZD', 'MEANPOT', 'MEANSHR', 'SHRGT45', 'R_VALUE', 'AREA_ACR']
        df_filtered = df[(df[key_cols] != 0).all(axis=1)].copy()
        df_filtered['T_REC'] = df_filtered['T_REC'].apply(parse_tai_string)

        # Loading the Y_train data
        client = hek.HEKClient()
        flare_data = client.search(
            a.Time(start_time, end_time),
            a.hek.FL,
        )

        names = [name for name in flare_data.colnames if len(flare_data[name].shape) <= 1]
        flare_data_flat = flare_data[names]
        flare_data_df = flare_data_flat.to_pandas()

        flare_data = flare_data_df[['ar_noaanum', 'fl_goescls', 'event_peaktime']]
        flare_data = flare_data[
            (flare_data['ar_noaanum'] > 0) & (flare_data['fl_goescls'] != '')]

        df_filtered['flare_label'] = "None"

        for idx, flare in flare_data.iterrows():
            ar = flare['ar_noaanum']
            flare_time = flare['event_peaktime']
            flare_class = flare['fl_goescls']

            match = (df_filtered['NOAA_AR'] == ar) & (df_filtered['T_REC'] <= flare_time) & (flare_time <= df_filtered['T_REC'] + pd.Timedelta(hours=24))

            label = flare_class
            df_filtered.loc[match, 'flare_label'] = label

        # Append to combined DataFrame
        combined_df = pd.concat([combined_df, df_filtered], ignore_index=True)

        print(f"Day: {i + 1}/{number_of_days} processed")

    # Save the combined DataFrame to CSV
    combined_df.to_csv(csv_path, index=False)

    print(combined_df)

data_loader(30)
