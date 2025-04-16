from .data_loader import hmi_data_loader
import argparse

def active_region_detector(start_time, end_time=None):
    """ 
    This function detects active regions in solar images.
    It uses a helper function to identify and classify active regions.
    The function returns the detected active regions and their specifications.

    Parameters:
    start_time (str): Start time for the data collection in YYYY-MM-DD HH:MM format.
    end_time (str, optional): End time for the data collection in YYYY-MM-DD HH:MM format.

    Returns:
    Data of the detected active regions(coordinates, area, mean magnetic field strength, etc.), 
    as well as a graph of the active regions with bounding boxes
    """
    active_regions = hmi_data_loader(start_time, end_time)

    return active_regions

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Active Region Detector")
    parser.add_argument("--start_time", type=str, required=True, help="Start time in YYYY-MM-DD HH:MM format")
    parser.add_argument("--end_time", type=str, help="End time(Optional) in YYYY-MM-DD HH:MM format")

    args = parser.parse_args()
    active_region_detector(args.start_time, args.end_time)