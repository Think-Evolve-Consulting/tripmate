import logging
import numpy as np
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

def cleanse_gps_data(
    trips_info_df, 
    trips_gps_df
):
    logger.info("Cleansing trips info and gps data ...")

    # delete trips that do not have any gps pings data
    logger.debug("Deleting trips that do not have any gps pings data")
    trips_missing_pings = \
        trips_info_df[~(trips_info_df.TripLogId.isin(trips_gps_df.TripLogId.unique()))].index
    trips_info_df.drop(index=trips_missing_pings, inplace=True)
    logger.debug(f"Deleted {len(trips_missing_pings)} trips that do not have any gps pings data")

        # anonymize DumperMachineName
    machine_name_dict = {
        'Mercedes Arocs (DR67820)': '101', 
        'Volvo A45G FS (3834) 12323834': '201', 
        'Scania R580 (AJ90818)': '102', 
        'Mercedes (SD89781) 2763': '103', 
        'Scania R590 (AJ94392) AJ94392': '104', 
        'Scania R580 (PD 70495)': '105', 
        'Scania R580 (AJ91132)': '106', 
        'Mercedes Arocs (SD95898) 2902': '106', 
        'Scania R540 AJ94080': '107', 
        'Scania 590 (AJ94391)': '108', 
        'Cat 745 B ( 1484 ) 12321484': '202', 
        'Scania R 580 (PD 69849)': '109', 
        'SCANIA R490 8x4 4AKSLET 2505': '203', 
        'SCANIA R 520 (PD 69848)': '110', 
        'A45 FS (3834) 12323834': '201', 
        'PD 69848': '110', 
        'Volvo A45 (4060) 12324060': '204', 
        'Scania R590 AJ94391': '108', 
        'Scania R580 AJ91826': '109', 
        'NA': np.nan
        }
    trips_info_df["DumperMachineName"] = \
        [machine_name_dict[x.strip()] for x in trips_info_df.DumperMachineName.fillna('NA')]
    trips_info_df["DumperMachineName"] = trips_info_df.DumperMachineName.astype(float)
    logger.debug("Anonymized DumperMachineName in trips info data")

    # update missing DumperMachineNumbers with anonymized DumperMachineName data
    trips_info_df.DumperMachineNumber.fillna(trips_info_df.DumperMachineName, inplace=True)
    logger.debug("Updated missing DumperMachineNumber with anonymized DumperMachineName data")

    # drop DumperMachineName as it contains potential PII
    trips_info_df.drop(columns=['DumperMachineName'], inplace=True)

    # delete exact duplicates in trips gps data
    logger.debug("De-duplicating trips gps data ...")
    logger.debug(f"Number of samples = {trips_gps_df.shape[0]}")
    logger.debug(f"Number of duplicated samples (all columns considered) = {trips_gps_df.duplicated().sum()}")
    trips_gps_df.drop_duplicates(keep="first", ignore_index=True, inplace=True)
    logger.debug("Exact duplicates were cleansed by retaining just one copy of the duplicates")
    logger.debug(f"Number of samples after de-duplication = {trips_gps_df.shape[0]}")
    logger.debug("De-duplication of trips gps data completed successfully")

    # change format of Timestamp in trips gps data from str (object) to datetime
    trips_gps_df.rename(columns={'Timestamp': 'OriginalTimestamp'}, inplace=True)
    try:
        logger.debug("Attempting to convert timestamp in gps data to datetime format using default option ...")
        trips_gps_df['Timestamp'] = pd.to_datetime(trips_gps_df['OriginalTimestamp'])
        logger.debug("Successfully converted timestamp in gps data to datetime format")
    except:
        logger.debug("Attempting to convert timestamp in gps data to datetime format using ISO8601 option ...")
        trips_gps_df['Timestamp'] = pd.to_datetime(trips_gps_df['OriginalTimestamp'], format='ISO8601')
        logger.debug("Successfully converted timestamp in gps data to datetime format")

    return trips_info_df, trips_gps_df

def exclude_problematic_trips(
    trips_info_df, 
    trips_gps_df, 
    exclusion_candidates
    ):
    logger.info("Excluding trips info and gps data that can cause downstream issues ...")

    # delete gps pings and trips based on exclusion list
    exclusion_pings = \
        trips_gps_df[(trips_gps_df.TripLogId.isin(exclusion_candidates))].index
    trips_gps_df.drop(index=exclusion_pings, inplace=True)
    logger.debug(f"Deleted {len(exclusion_pings)} gps pings")

    exclusion_trips = \
        trips_info_df[(trips_info_df.TripLogId.isin(exclusion_candidates))].index
    trips_info_df.drop(index=exclusion_trips, inplace=True)
    logger.debug(f"Deleted {len(exclusion_trips)} trips from trips info")

    return trips_info_df, trips_gps_df

def cleanse_simplified_routes(
    routes_gdf 
    ):
    # delete unnecessary columns
    delete_columns = [
        'DumperMach', 'MachineTyp', 'LoadLongit', 'LoadLatitu', 'DumpLongit', 'DumpLatitu', 
        'MassTypeMa', 'Quantity', 'TripStart', 'TripEnd', 'Category', 'DataFile', 'DumperMa_1'
        ]
    for column in delete_columns:
        if column in routes_gdf.columns:
            routes_gdf.drop(columns=[column], inplace=True)
    logger.debug("Removed unnecessary columns from simplified routes geo data frame")

    return routes_gdf

def cleanse_trip_level_movement_params(
    trips_movement_params_df, 
    velocity_outlier_threshold=27, 
    acceleration_outlier_threshold=1.5
    ):
    logger.info("Cleansing trip level movement parameters ...")

    # drop rows that contain nan values
    trips_movement_params_df.dropna(inplace=True)
    logger.info(f"Number of trips with movement parameters after dropping NaNs: {trips_movement_params_df.shape[0]}")

    # drop outliers based on velocity / acceleration / deceleration
    trips_movement_params_df.drop(
        trips_movement_params_df.query(f"VelocityMean >= {velocity_outlier_threshold}").index, 
        inplace=True
        )
    trips_movement_params_df.drop(
        trips_movement_params_df.query(f"AccelerationMean >= {acceleration_outlier_threshold}").index, 
        inplace=True
        )
    trips_movement_params_df.drop(
        trips_movement_params_df.query(f"DecelerationMean >= {acceleration_outlier_threshold}").index, 
        inplace=True
        )
    logger.info(f"Number of trips with movement parameters after dropping outliers: {trips_movement_params_df.shape[0]}")

    return trips_movement_params_df