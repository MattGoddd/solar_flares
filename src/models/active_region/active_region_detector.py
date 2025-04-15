from .data_loader import hmi_data_loader, hmi_image_viewer

def active_region_detector(start_time, end_time):
    """ 
    This function detects active regions in solar images.
    It uses a pre-trained model to identify and classify active regions.
    The function returns the detected active regions and their classifications.

    Parameters:
    start_time (str): Start time for the data collection in YYYY.MM.DD_HH:MM_TAI format.
    end_time (str): End time for the data collection in YYYY.MM.DD_HH:MM_TAI format.

    Returns:
    Coordinates of the detected active regions and their classifications, or maybe magnetogram data (ie, strength of magnetic field at that region)
    """


    