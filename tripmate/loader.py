import os
import glob
import shutil
import logging
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)

def load_gps_data(
    zip_file_path, 
    temp_dir, 
    unpack_zip=True
    ):
    logger.info("Loading trips info and gps data ...")

    # check if zip file exists
    if not os.path.exists(zip_file_path) or not os.path.isfile(zip_file_path):
        raise Exception(f"Trips data (info and gps) zip file does not exist. File path: {zip_file_path}")
    
    # check if temp directory exists
    if not os.path.exists(temp_dir):
        raise Exception(f"Temp directory does not exist. File path: {temp_dir}")
    
    # unzip trips info and gps zip file
    if unpack_zip is True:
        shutil.unpack_archive(
            filename=zip_file_path,
            extract_dir=temp_dir,
            format='zip'
            )
        logger.info(f"Extracted contents of zip file to {temp_dir}")
    else:
        logger.info(f"Expecting zip file to have been already extracted to {temp_dir}")
    
    # load trips info data from multiple csv files into a single data frame
    trip_csv_files = glob.glob(os.path.join(temp_dir, 'GPSData', 'tripsInfo') + "/*.csv")
    trip_csv_files.sort()
    trips_info_df = pd.concat(map(pd.read_csv, trip_csv_files), ignore_index=True)
    logger.info(f"Consolidated trips info data: {trips_info_df.shape[0]} rows loaded")

    # load trips info data from multiple csv files into a single data frame
    gps_csv_files = glob.glob(os.path.join(temp_dir, 'GPSData', 'trips') + "/*.csv")
    gps_csv_files.sort()
    trips_gps_df = pd.concat(map(pd.read_csv, gps_csv_files), ignore_index=True)
    logger.info(f"Consolidated trips gps data: {trips_gps_df.shape[0]} rows loaded")

    logger.info("Loaded trips info and gps data successfully")

    return trips_info_df, trips_gps_df

def load_simplified_routes(
    shapefile_path
    ):
    # load shapefile contents
    routes_gdf = gpd.read_file(shapefile_path)
    return routes_gdf

def load_trip_level_movement_params(
    file_path
    ):
    logger.info("Loading trip level movement parameters ...")
    trips_movement_params_df = pd.DataFrame({})

    # read trip level movement parameters summary from given file path
    if os.path.exists(file_path) and os.path.isfile(file_path):
        trips_movement_params_df = \
            pd.read_csv(
                file_path,
                header="infer",
                parse_dates=["TripStart", "TripEnd"],
                infer_datetime_format=True
                )
        logger.info(f"Loaded trip level movement parameters for {trips_movement_params_df.shape[0]} trips")
    else:
        raise Exception(f"Unable to load data from {file_path}")

    return trips_movement_params_df