import pandas as pd
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
import astropy.units as u
from datetime import datetime
from sunpy.net.jsoc import JSOCClient

def active_region_data_loader(start_time, end_time = None)  -> pd.DataFrame:
    """
    This function takes in a Dataframe containing the coordinates of active regions,
    and returns the physical features of these active regions with respect to these active regions

    Parameters:
    df: DataFrame containing active region data.

    Return:
    DataFrame with physical features of active regions.

    """
    # if not df:
    #     raise ValueError("The DataFrame is empty. Please provide a valid DataFrame.")
     
    # result = Fido.search(
    #     a.Time(start_time, end_time),
    #     a.Instrument.hmi,
    #     a.Physobs.los_magnetic_field,
    #     a.Provider.jsoc,
    #     a.Sample(12 * u.minute),
    #     a.jsoc.Series('hmi.sharp_cea_720s'),
    # )
    # if result:
    #     sharp_file = Fido.fetch(result)
    # else:
    #     sharp_file = None
    # if not sharp_file:
    #         raise ValueError(f"No data found for the specified time range: {start_time} to {end_time}.")
    client = JSOCClient()
    print(client.help())
    
    # for region in df:
    #     hp_coord = region['hp_coord']
    #     hg_coord = region['hg_coord']
    #     timestamp = region['timestamp']
    #     sharp_file = Fido.search(
    #         a.Time(start_time, end_time),
    #         a.Instrument.hmi,
    #         a.Physobs.magnetic_field,
    #         a.Provider.jsoc,
    #         a.Sample(12 * u.minute),
    #         a.jsoc.series('hmi.B_720s'),
    #     )
    #     if not sharp_file:
    #         raise ValueError(f"No data found for the specified time range: {start_time} to {end_time}.")
        

active_region_data_loader("2025-04-15 21:55", "2025-04-15 21:55")