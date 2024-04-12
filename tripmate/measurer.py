import logging
import numpy as np
import pandas as pd
from pyproj import Geod

logger = logging.getLogger(__name__)

def get_inter_ping_distance(
    trips_gps_df, 
    level, 
    ellipsoid="WGS84"
):
    # validate the option provided for level
    if level == 'TripLogId':
        compute_level = 'trip'
    elif level == 'DumperMachineNumber':
        compute_level = 'machine'
    else:
        raise Exception(
            f"Invalid option for level: {level}." 
            "Allowed options: [TripLogId, DumperMachineNumber]"
            )

    logger.info(f"Computing geodesic distance between successive pings at {compute_level} level ...")

    # get time and location coordinates previous ping
    trips_gps_df["PreviousLatitude"] = trips_gps_df.groupby(level)["Latitude"].shift()
    trips_gps_df["PreviousLongitude"] = trips_gps_df.groupby(level)["Longitude"].shift()

    # for the first ping of a trip or machine, there will not be a previous ping
    # set the previous ping details to be same as the current ping details
    trips_gps_df.PreviousLatitude.fillna(trips_gps_df.Latitude, inplace=True)
    trips_gps_df.PreviousLongitude.fillna(trips_gps_df.Longitude, inplace=True)

    logger.debug("Collected previous ping details")

    # compute geodesic distance between two successive pings
    geod = Geod(ellps=ellipsoid)
    _, _, incremental_distances = \
        geod.inv(
            lons1=trips_gps_df.PreviousLongitude,
            lats1=trips_gps_df.PreviousLatitude,
            lons2=trips_gps_df.Longitude,
            lats2=trips_gps_df.Latitude
            )
    logger.debug("Computed geodesic distance between successive pings")

    return incremental_distances

def get_movement_params(
    gps_df, 
    level, 
    resampling_interval=None
):
    # if resampling interval is not provided, compute elapsed time between successive trips
    if resampling_interval is None:
        logger.info("Computing movement parameters by computing time elapsed ...")
        # compute elapsed time between two successive pings
        gps_df["PreviousTime"] = gps_df.groupby(level)["Timestamp"].shift()
        gps_df.PreviousTime.fillna(gps_df.Timestamp, inplace=True)
        gps_df["TimeElapsed"] = \
            (gps_df.Timestamp - gps_df.PreviousTime).dt.total_seconds()
    else:
        logger.debug(f"Computing movement parameters as per given resampling interval ...")
        gps_df["TimeElapsed"] = resampling_interval

    # compute incremental distance based on cumulative distance
    gps_df["PreviousDistance"] = gps_df.groupby(level)["CumulativeDistance"].shift().fillna(0)
    gps_df["DistanceIncrement"] = gps_df.CumulativeDistance - gps_df.PreviousDistance
    logger.debug("Computed distance increment between successive data points")
        
    # compute velocity and acceleration
    gps_df["Velocity"] = gps_df.DistanceIncrement.div(gps_df["TimeElapsed"])
    gps_df["Acceleration"] = \
        (gps_df.Velocity
            - gps_df.groupby(level)["Velocity"].shift().fillna(0)
            ) / gps_df["TimeElapsed"]
    logger.debug("Computed velocity and acceleration")
    
    # compute derived parameters using velocity and acceleration
    gps_df["RelativePositiveAcceleration"] = \
        (gps_df[gps_df["Acceleration"]>0]["Velocity"]
         * gps_df[gps_df["Acceleration"]>0]["Acceleration"]
         * gps_df[gps_df["Acceleration"]>0]["TimeElapsed"])
    
    gps_df["VASquared"] = \
        gps_df["Velocity"] * (gps_df["Acceleration"]**2)

    # handle nan values
    gps_df.fillna(
        {
            'Velocity': 0.0, 
            'Acceleration': 0.0, 
            'VASquared': 0.0, 
            'RelativePositiveAcceleration': 0.0
            }, 
        inplace=True
        )

    logger.info("Computed movement parameters successfully")

    return gps_df

def summarize_trip_movement_params(
    movement_params_df,
    idling_velocity_threshold=1*1000/3600,
    acceleration_threshold=0.1, 
    deceleration_threshold=-0.1, 
    fast_acceleration_threshold=1.5
):
    trip_distance = movement_params_df.DistanceIncrement.sum()
    velocity_mean = movement_params_df.Velocity.mean()
    velocity_sd = movement_params_df.Velocity.std()
    velocity_except_idle_mean = \
        movement_params_df.query(
            f"Velocity > {idling_velocity_threshold}"
            ).Velocity.mean()
    velocity_except_idle_sd = \
        movement_params_df.query(
            f"Velocity > {idling_velocity_threshold}"
            ).Velocity.std()

    acceleration_mean = \
        movement_params_df.query(
            f"Acceleration > {acceleration_threshold}"
            ).Acceleration.mean()
    acceleration_sd = \
        movement_params_df.query(
            f"Acceleration > {acceleration_threshold}"
            ).Acceleration.std()
    acceleration_percent = \
        movement_params_df.query(
            f"Acceleration > {acceleration_threshold}"
            ).Acceleration.count() / movement_params_df.shape[0]

    deceleration_mean = \
        np.abs(
            movement_params_df.query(
                f"Acceleration < {deceleration_threshold}"
                ).Acceleration
                ).mean()
    deceleration_sd = \
        np.abs(
            movement_params_df.query(
                f"Acceleration < {deceleration_threshold}"
                ).Acceleration
                ).std()
    deceleration_percent = \
        movement_params_df.query(
            f"Acceleration < {deceleration_threshold}"
            ).Acceleration.count() / movement_params_df.shape[0]

    idling_percent = \
        movement_params_df.query(f"Acceleration >= {deceleration_threshold}")\
                        .query(f"Acceleration <= {acceleration_threshold}")\
                        .query(f"Velocity <= {idling_velocity_threshold}")\
                        .Acceleration.count() / movement_params_df.shape[0]
    cruising_percent = \
        1.00 - (acceleration_percent + deceleration_percent  + idling_percent)

    fast_accelerations = \
        movement_params_df.query(f"abs(Acceleration) >= {fast_acceleration_threshold}")\
                        .Acceleration.count()

    relative_positive_acceleration_mean = \
        (
            movement_params_df.query("Acceleration > 0").RelativePositiveAcceleration\
        / trip_distance
        ).mean()

    relative_positive_acceleration_sd = \
        (
            movement_params_df.query("Acceleration > 0").RelativePositiveAcceleration\
        / trip_distance
        ).std()

    v_a_squared_mean = movement_params_df.VASquared.mean()
    v_a_squared_sd = movement_params_df.VASquared.std()

    return dict(
        TripDistance=trip_distance, 
        VelocityMean=velocity_mean, 
        VelocitySD=velocity_sd, 
        VelocityExIdleMean=velocity_except_idle_mean, 
        VelocityExIdleSD=velocity_except_idle_sd, 
        AccelerationMean=acceleration_mean, 
        AccelerationSD=acceleration_sd, 
        DecelerationMean=deceleration_mean, 
        DecelerationSD=deceleration_sd, 
        AccelerationTimePercent=acceleration_percent, 
        DecelerationTimePercent=deceleration_percent, 
        IdlingTimePercent=idling_percent, 
        CruisingTimePercent=cruising_percent, 
        FastAccelerationCount=fast_accelerations, 
        RelativePositiveAccelerationMean=relative_positive_acceleration_mean, 
        RelativePositiveAccelerationSD=relative_positive_acceleration_sd, 
        VASquaredMean=v_a_squared_mean, 
        VASquaredSD=v_a_squared_sd
        )

def summarize_trips_movement_params(
    movement_params_df, 
    trips_info_df, 
    idling_velocity_threshold=1*1000/3600,
    acceleration_threshold=0.1, 
    deceleration_threshold=-0.1, 
    fast_acceleration_threshold=1.5
):
    # movement_params_df may contain multiple trips
    # process each trip separately, summarize movement parameters and save in a data frame

    # create a data frame with trip info data for given trips; 
    # later on, add columns for summarized movement parameters
    trips_movement_params_df = \
        trips_info_df\
            .query(f"TripLogId.isin({movement_params_df.TripLogId.unique().tolist()})")\
            [[
                "TripLogId",
                "DumperMachineNumber",
                "MachineType",
                "MassTypeMaterial",
                "Quantity",
                "TripStart",
                "TripEnd",
                "TripDateString",
                "TripName"
                ]].copy()
    
    for index, row in trips_movement_params_df.iterrows():
        # call the 'single' trip movement parameters summarization function
        mp_dict = \
            summarize_trip_movement_params(
                movement_params_df=\
                    movement_params_df\
                        .query(f"TripLogId == '{row.TripLogId}'")\
                        .query(f"ActivityClass == 'DRIVE'"),
                idling_velocity_threshold=idling_velocity_threshold,
                acceleration_threshold=acceleration_threshold, 
                deceleration_threshold=deceleration_threshold, 
                fast_acceleration_threshold=fast_acceleration_threshold
            )
        
        # update summary statistics of trip level movement parameters to data frame
        trips_movement_params_df.loc[index, mp_dict.keys()] = mp_dict
    
    return trips_movement_params_df