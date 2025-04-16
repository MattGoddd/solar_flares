from .data_loader import hmi_data_loader
import argparse

def active_region_detector(start_time, end_time=None):
    """ 
    This function detects active regions in solar images.
    It uses a pre-trained model to identify and classify active regions.
    The function returns the detected active regions and their classifications.

    Parameters:
    start_time (str): Start time for the data collection in YYYY-MM-DD HH:MM format.
    end_time (str): End time for the data collection in YYYY-MM-DD HH:MM format.

    Returns:
    Coordinates of the detected active regions and their classifications, or maybe magnetogram data (ie, strength of magnetic field at that region)
    """
    active_regions = hmi_data_loader(start_time, end_time)

    return active_regions

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Active Region Detector")
    parser.add_argument("--start_time", type=str, required=True, help="Start time in YYYY-MM-DD HH:MM format")
    parser.add_argument("--end_time", type=str, help="End time(Optional) in YYYY-MM-DD HH:MM format")

    args = parser.parse_args()
    active_region_detector(args.start_time, args.end_time)