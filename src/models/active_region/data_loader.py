import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.net import Fido, attrs as a, hek
from sunpy.time import parse_time
from sunpy.net.hek import HEKClient
from sunpy.net.hek.attrs import EventType, OBS
import sunpy.map
import numpy as np
from skimage.measure import label, regionprops
import pandas as pd
import yaml


def hmi_data_loader(start_time, end_time=None):
    """
    Download HMI data from the Solar Dynamics Observatory (SDO) using SunPy.
    Images are taken in 12min intervals, and the data is filtered to find active regions.
    It is recommended to put a time frame of 12 minutes for the data collection to obtain at least one result, 
    Or to leave end_time blank which sets the end time to 12 minutes after the start time.
    The function returns the data of the active regions in the given time frames.

    Parameters:
    start_time: Start time for the data collection in YYYY-MM-DD HH:MM format.
    end_time (Optional): End time for the data collection in YYYY-MM-DD HH:MM format.

    Return:
    Data for the active regions in the given time frames in a list
    """
    start_time_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")

    if end_time:
        end_time_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
    else:
        end_time_dt = start_time_dt + timedelta(minutes=12)

    print(start_time_dt, type(start_time_dt), end_time_dt, type(end_time_dt))
    
    start_time = start_time_dt.strftime("%Y-%m-%dT%H:%M:%S")
    end_time = end_time_dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Parameters

    with open("config/active_region_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    download_dir = Path(config["download_dir"])

    # Creates the download directory if it doesn't exist

    os.makedirs(download_dir, exist_ok=True)

    # Threshold for magnetic field strength in Gauss

    threshold = config["threshold"]

    # Minimum area in pixels for a region to be considered

    min_area = config["min_area"]

    # Set the boundaries for the bounding boxes in the image and for filtered data

    max_reasonable_box_size = config["max_reasonable_box_size"]

    # To set if the desired data is to be saved into a CSV file or not

    csv_save = config["csv_save"]
    print(start_time, end_time)

    filtered_region_data = []

    counter = 0

    # Define the HMI data query
    query = Fido.search(
        a.Time(start_time, end_time), 
        a.jsoc.Series("hmi.M_720s_nrt"),
        a.jsoc.Notify("mattgoh2004@gmail.com"),
        )
    print(query)

    files = Fido.fetch(query, path=download_dir)
    for file in files:
    
        try:
            print(f"Processing file: {file}")
            hmi_map = sunpy.map.Map(file)
            mag_field = hmi_map.data.astype(np.float32)

            #Threshold values for magnetic field

            mask = np.abs(mag_field) > threshold
            labeled = label(mask)
            props = regionprops(labeled)
            timestamp = hmi_map.date.isot
            region_data = []

            for region in props:
                if region.area > min_area:
                    minr, minc, maxr, maxc = region.bbox
                    # Calculate the center of the region
                    cy, cx = region.centroid
                    # Calculate the area of the region in pixels
                    area = region.area
                    # Calculate the mean magnetic field strength in the region
                    mean_strength = np.mean(mag_field[labeled == region.label])
                    # Append the data to the list
                    region_data.append({
                        'timestamp': timestamp,
                        'center_x': cx,
                        'center_y': cy,
                        'hp_coord': hmi_map.pixel_to_world(cx * u.pix, cy * u.pix),
                        'hg_coord': hmi_map.pixel_to_world(cx * u.pix, cy * u.pix).heliographic_stonyhurst,
                        'area': area,
                        'mean_strength': mean_strength,
                        'bbox_xmin': minc,
                        'bbox_xmax': maxc,
                        'bbox_ymin': minr,
                        'bbox_ymax': maxr,
                    })

            # Save the region data to a CSV file

            print("region data processed")

            if csv_save == True:
                base_filename = os.path.basename(file).replace('.fits', '.csv')
                df = pd.DataFrame(region_data)
                df.to_csv(os.path.join(download_dir, base_filename), index=False)
                print(f"Saved region data to {base_filename}")

            # Plot the bounding boxes for each labeled region
            plt.figure(figsize=(10, 8))
            plt.imshow(mag_field, cmap='seismic', origin='upper', vmin=-2000, vmax=2000)
            plt.colorbar(label='Magnetic Field Strength (Gauss)')
            plt.title('HMI Magnetogram with Active Regions Labeled  ')

            # Overlay bounding boxes for each active region
            for region in region_data:
                xmin, xmax = region['bbox_xmin'], region['bbox_xmax']
                ymin, ymax = region['bbox_ymin'], region['bbox_ymax']
                width = xmax - xmin
                height = ymax - ymin
                if width < max_reasonable_box_size and height < max_reasonable_box_size:
                    plt.gca().add_patch(plt.Rectangle(
                        (xmin, ymin), width, height,
                        edgecolor='yellow', facecolor='none', linewidth=1.5
                        ))
                    filtered_region_data.append(region)
                    counter += 1
                else:
                    print(f"Skipping region with large bounding box: Region {region['label']}")

            plt.show()


            # Deletes the FITS file after processing

            os.remove(file)
            print(f"Deleted file: {file}")

            print("Processing next file!")

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    
    print(f"Finished processing files for {start_time} to {end_time}, with total number of regions: {counter}")

    filtered_region_data = pd.DataFrame(filtered_region_data)

    print(filtered_region_data)

    return filtered_region_data

hmi_data_loader("2025-04-15 21:55")

# def active_region_numbering(start_time, end_time, threshold = 5, df: pd.DataFrame = None) -> pd.DataFrame:
#     """
#     Establishes the NOAA AR numbering of the active regions indentified based on a certain threshold value

#     Parameters:
#     df: DataFrame containing active region data.

#     Return:
#     DataFrame with numbered active regions.
#     """
#     client = hek.HEKClient()

#     with open("config/active_region_config.yaml", "r") as config_file:
#         config = yaml.safe_load(config_file)

#     start_time = parse_time(start_time)
#     end_time = parse_time(end_time)

#     noaa_list = client.search (
#         a.Time(start_time, end_time),
#         EventType("AR"),
#     )

#     names = [name for name in noaa_list.colnames if noaa_list[name].ndim <= 1]
#     df_noaa = noaa_list[names].to_pandas()

#     df_noaa.to_csv(os.path.join(Path(config["download_dir"]), "active_regions.csv"), index=False)

#     # Just show key columns
#     print(df_noaa)


# # active_region_numbering(start_time="2025-04-15 21:55", end_time="2025-04-15 21:55")
    

# def hmi_image_viewer(file):
#     """
#     View HMI image using matplotlib and overlay bounding boxes for active regions.

#     Parameters:
#     file: Path to the HMI FITS file.

#     Return:
#     Displays the HMI image with bounding boxes around active regions.
#     """

#     # Open the FITS file
#     with fits.open(file) as hdul:
#         # Get the data from the first extension
#         print(hdul.info())
#         data = hdul[1].data
#         header = hdul[1].header
#         bscale = header.get('BSCALE', 1)
#         bzero = header.get('BZERO', 0)

#         mag_field = data * bscale + bzero

#         # Threshold for magnetic field strength in Gauss
#         threshold = 100

#         # Create a mask for regions with magnetic field strength above the threshold
#         mask = np.abs(mag_field) > threshold

#         # Label connected regions
#         labeled, num_features = label(mask)

#         # Plot the data
#         plt.figure(figsize=(10, 8))
#         plt.imshow(mag_field, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
#         plt.colorbar(label='Magnetic Field Strength (Gauss)')
#         plt.title('HMI Magnetogram with Active Regions')

#         # Overlay bounding boxes for each labeled region
#         for label_id in range(1, num_features + 1):
#             coords = np.argwhere(labeled == label_id)
#             if coords.shape[0] < 1000:  # Minimum area threshold
#                 continue
#             y_min, x_min = coords.min(axis=0)
#             y_max, x_max = coords.max(axis=0)
#             plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
#                                               edgecolor='yellow', facecolor='none', linewidth=1.5))

#         plt.show()


    

# # hmi_image_viewer(r"C:\Users\UserAdmin\dsta_project\solar_flares\data\hmi.M_720s_files\hmi.M_720s.20250404_191200_TAI.3.magnetogram.fits")





