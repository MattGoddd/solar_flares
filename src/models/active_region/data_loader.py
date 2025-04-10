import os
import time
import requests
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits

def hmi_data_loader(start_time, end_time):
    """
    Load HMI data from JSOC API.

    Parameters:
    start_time, end_time: YYYY.MM.DD_HH:MM_TAI

    Returns:
    Data Uploaded in solar/flares/data/hmi.M_720s_files
    """
    # --- Setting up --- #

    time_frame = f"{start_time}-{end_time}"

    record_set = f"hmi.M_720s[{time_frame}]"

    segment = "magnetogram"

    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # Get the path to the project root directory (solar_flares)

    output_dir = project_root / "data" / "hmi.M_720s_files"

    # --- Submitting Export Request ---#

    export_url = "http://jsoc.stanford.edu/cgi-bin/ajax/export.cgi"
    export_params = {
        'ds': record_set,
        'seg': segment,
        'format': 'fits',
        'requestor': 'anonymous',
        'protocol': 'url',
    }

    print(f"Submitting export request for {record_set}...")
    
    resp = requests.get(export_url, params = export_params)

    data = resp.json()

    if data['status'] != 'success':
        print("Error submitting request:", data.get('errmsg', 'Unknown error'))
        exit()
    
    request_id = data['requestid']

    status_url = f"http://jsoc.stanford.edu/cgi-bin/ajax/export.cgi?op=exp_status&requestid={request_id}"

    if request_id:
        print("success")

def hmi_image_viewer(file):
    """
    View HMI image using matplotlib.

    Parameters:
    file: Path to the HMI FITS file.
    """
    import matplotlib.pyplot as plt
    from astropy.io import fits

    # Open the FITS file
    with fits.open(file) as hdul:
        # Get the data from the first extension
        print(hdul.info())
        data = hdul[1].data
        header = hdul[1].header
        bscale = header.get('BSCALE', 1)
        bzero = header.get('BZERO', 0)

        mag_field = data * bscale + bzero

        hdul.close()


        # Plot the data
        plt.imshow(mag_field, cmap='seismic', origin='lower', vmin=-2000, vmax=2000)
        plt.colorbar(label='Magnetic Field Strength (Gauss)')
        plt.title('HMI Magnetogram (Seismic Colormap)')
        plt.show()	

hmi_image_viewer(r"C:\Users\UserAdmin\dsta_project\solar_flares\data\hmi.M_720s_files\hmi.M_720s.20250404_191200_TAI.3.magnetogram.fits")





