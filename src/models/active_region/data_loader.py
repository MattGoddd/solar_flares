import os
from datetime import datetime, timedelta
import requests
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from sunpy.net import Fido, attrs as a
import sunpy.map
import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label, regionprops
import pandas as pd


def hmi_data_loader(start_time, end_time=None):
    """
    Download HMI data from the Solar Dynamics Observatory (SDO) using SunPy.

    Parameters:
    start_time: Start time for the data collection in YYYY.MM.DD_HH:MM_TAI format.
    end_time: End time for the data collection in YYYY.MM.DD_HH:MM_TAI format.

    Return:
    List of data in a database file 
    """

    if end_time is None:
        start_time_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        end_time_dt = start_time_dt + timedelta(minutes=12)
        end_time = end_time_dt.strftime("%Y-%m-%d %H:%M")

    # Parameters

    download_dir = Path("hmi.M_720s_files")

    # Creates the download directory if it doesn't exist

    os.makedirs(download_dir, exist_ok=True)

    # Threshold for magnetic field strength in Gauss

    threshold = 100

    # Minimum area in pixels for a region to be considered

    min_area = 1000

    # Define the HMI data query
    query = Fido.search(
        a.Time(start_time, end_time), 
        a.Instrument.hmi, 
        a.Physobs.los_magnetic_field,
        a.Sample(720 * u.s)
        )

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
                        'area': area,
                        'mean_strength': mean_strength,
                        'label': region.label,
                        'bbox_xmin': minc,
                        'bbox_xmax': maxc,
                        'bbox_ymin': minr,
                        'bbox_ymax': maxr,
                    })

            # Save the region data to a CSV file

            print("region data processed")

            base_filename = os.path.basename(file).replace('.fits', '.csv')
            df = pd.DataFrame(region_data)
            df.to_csv(os.path.join(download_dir, base_filename), index=False)
            print(f"Saved region data to {base_filename}")

            # Plot the bounding boxes for each labeled region
            plt.figure(figsize=(10, 8))
            plt.imshow(mag_field, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
            plt.colorbar(label='Magnetic Field Strength (Gauss)')
            plt.title('HMI Magnetogram with Active Regions')

            # Overlay bounding boxes for each active region
            for region in region_data:
                xmin, xmax = region['bbox_xmin'], region['bbox_xmax']
                ymin, ymax = region['bbox_ymin'], region['bbox_ymax']
                width = xmax - xmin
                height = ymax - ymin
                max_reasonable_box_size = 1000  # Maximum reasonable box size in pixels
                if width < max_reasonable_box_size and height < max_reasonable_box_size:
                    plt.gca().add_patch(plt.Rectangle(
                        (xmin, ymin), width, height,
                        edgecolor='yellow', facecolor='none', linewidth=1.5
                        ))

            plt.show()


            # Deletes the FITS file after processing
            print(f"Processed and saved: {base_filename} ({len(region_data)} regions)")


        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
        print("finished processing files")

hmi_data_loader("2025-04-04 21:55")



def hmi_image_viewer(file):
    """
    View HMI image using matplotlib and overlay bounding boxes for active regions.

    Parameters:
    file: Path to the HMI FITS file.

    Return:
    Displays the HMI image with bounding boxes around active regions.
    """

    # Open the FITS file
    with fits.open(file) as hdul:
        # Get the data from the first extension
        print(hdul.info())
        data = hdul[1].data
        header = hdul[1].header
        bscale = header.get('BSCALE', 1)
        bzero = header.get('BZERO', 0)

        mag_field = data * bscale + bzero

        # Threshold for magnetic field strength in Gauss
        threshold = 100

        # Create a mask for regions with magnetic field strength above the threshold
        mask = np.abs(mag_field) > threshold

        # Label connected regions
        labeled, num_features = label(mask)

        # Plot the data
        plt.figure(figsize=(10, 8))
        plt.imshow(mag_field, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
        plt.colorbar(label='Magnetic Field Strength (Gauss)')
        plt.title('HMI Magnetogram with Active Regions')

        # Overlay bounding boxes for each labeled region
        for label_id in range(1, num_features + 1):
            coords = np.argwhere(labeled == label_id)
            if coords.shape[0] < 1000:  # Minimum area threshold
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              edgecolor='yellow', facecolor='none', linewidth=1.5))

        plt.show()


    

# hmi_image_viewer(r"C:\Users\UserAdmin\dsta_project\solar_flares\data\hmi.M_720s_files\hmi.M_720s.20250404_191200_TAI.3.magnetogram.fits")





