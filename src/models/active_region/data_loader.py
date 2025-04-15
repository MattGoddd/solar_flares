import os
import time
import requests
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
import astropy.units as u
from sunpy.net import Fido, attrs as a
import sunpy.map
import numpy as np
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops
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
        end_time = start_time

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
            labeled, num_features = label(mask)
            timestamp = hmi_map.date.isot
            labeled = label(mask)
            regions = regionprops(labeled)

            # Plot the image
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(mag_field, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
            plt.colorbar(im, ax=ax, label="Magnetic Field Strength (Gauss)")
            ax.set_title("HMI Magnetogram with Active Region Bounding Boxes")

            # Draw bounding boxes
            for i, region in enumerate(regions):
                if region.area < 1000:  # skip small regions
                    continue

                # region.bbox = (min_row, min_col, max_row, max_col)
                minr, minc, maxr, maxc = region.bbox
                width = maxc - minc
                height = maxr - minr

                rect = patches.Rectangle(
                    (minc, minr), width, height,
                    linewidth=2, edgecolor='yellow', facecolor='none'
                )
                ax.add_patch(rect)

                # Optionally: label the region
                cy, cx = region.centroid
                ax.text(cx, cy, f"{i+1}", color='yellow', fontsize=8, ha='center')

            plt.show()
            region_data = []


            #Issue is here, the loop is too big
            for label_id in range(1, num_features + 1):
                coords = np.argwhere(labeled == label_id)

                if coords.shape[0] < min_area:
                    continue
                y_mean, x_mean = coords.mean(axis=0)

                region_data.append({
                    'timestamp': timestamp,
                    'label_id': label_id,
                    'centroid_x': x_mean,
                    'centroid_y': y_mean
                })
                print(label_id)

            # Save the region data to a CSV file

            print("region data processed")

            base_filename = os.path.basename(file).replace('.fits', '.csv')
            df = pd.DataFrame(region_data)
            df.to_csv(os.path.join(download_dir, base_filename), index=False)
            print(f"Saved region data to {base_filename}")

            # Deletes the FITS file after processing
            os.remove(file)
            print(f"Processed and saved: {base_filename} ({len(region_data)} regions)")


        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
        print("finished processing files")

# hmi_data_loader("2025-04-04 00:00", "2025-04-04 01:00")



def hmi_image_viewer(file):
    """
    View HMI image using matplotlib.

    Parameters:
    file: Path to the HMI FITS file.

    Return:
    Either shows images for each FITS file (if multiple, may be unfeasible) or returns the data.
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



        # Plot the data
        plt.imshow(mag_field, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
        plt.colorbar(label='Magnetic Field Strength (Gauss)')
        plt.title('HMI Magnetogram (Seismic Colormap)')
        plt.show()

        return hdul


    

hmi_image_viewer(r"C:\Users\UserAdmin\dsta_project\solar_flares\data\hmi.M_720s_files\hmi.M_720s.20250404_191200_TAI.3.magnetogram.fits")





