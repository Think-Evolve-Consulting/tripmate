import logging
import numpy as np
import pandas as pd
from scipy import interpolate

logger = logging.getLogger(__name__)

def get_interpolations(
    x_actuals,
    y_actuals,
    x_new_values,
    kind
):
    y_function = \
        interpolate.interp1d(
            x=x_actuals,
            y=y_actuals,
            kind=kind
            )
    y_new_values = y_function(x_new_values)
    return y_new_values

def get_resampled_gps_data(
    gps_df, 
    resampling_interval, 
    interpolation_column, 
    interpolation_kind='linear'
):
    # validate if the data provided belongs to a single machine
    unique_machines_count = gps_df.DumperMachineNumber.nunique()
    if unique_machines_count > 1:
        raise Exception("Resampling for data from multiple machines is not supported")
    else:
        machine_number = gps_df.DumperMachineNumber.unique()[0]

    logger.debug(f"Resampling gps data for machine number {int(machine_number)} ...")

    # calculate total duration (in seconds) covered by given gps data
    total_duration = \
        (gps_df.Timestamp.max() - gps_df.Timestamp.min()).total_seconds()
    logger.debug(f"Duration spanned by gps data: {total_duration} seconds")
    logger.debug(f"Minimum value of gps data: {gps_df.Timestamp.min()}")
    logger.debug(f"Maximum value of gps data: {gps_df.Timestamp.max()}")
    
    # get equally spaced timestamps as per resampling interval, total duration
    resampled_timestamps = \
        [gps_df.Timestamp.values.astype('datetime64[s]').min() + i
        for i in range(1, int(np.floor(total_duration))+1, resampling_interval)]
    resampled_timestamps = np.array(resampled_timestamps)
    logger.debug(f"Selected {resampled_timestamps.shape[0]-1} equally spaced time intervals")
    logger.debug(f"Minimum value of resampled timestamps: {np.min(resampled_timestamps)}")
    logger.debug(f"Maximum value of resampled timestamps: {np.max(resampled_timestamps)}")

    # create data frame and store the resampled timestamps
    resampled_gps_df = pd.DataFrame(dict(Timestamp=resampled_timestamps))
        
    # interpolate cumulative distance for resampled timestamps
    interpolations = \
        get_interpolations(
            x_actuals=gps_df.Timestamp.values.astype('datetime64[s]').astype('int64'), 
            y_actuals=gps_df[interpolation_column], 
            x_new_values=resampled_timestamps.astype('int64'), 
            kind=interpolation_kind
        )
    resampled_gps_df[interpolation_column] = interpolations
    resampled_gps_df['DumperMachineNumber'] = machine_number
    logger.debug(f"Interpolated {interpolation_column} using {interpolation_kind} method")

    # assign TripLogId for resampled data based on each trip's min and max gps ping timestamps
    for index, row in \
        gps_df\
            .groupby(['TripLogId'])\
            .agg(
                TripStart=('Timestamp', 'min'), 
                TripEnd=('Timestamp', 'max')
                )\
            .reset_index()\
            .iterrows():
        resampled_gps_df.loc[
            resampled_gps_df.Timestamp.between(
                row.TripStart.to_datetime64(), 
                row.TripEnd.to_datetime64()
                ), 
            'TripLogId'] = row.TripLogId

    # resampled data that belongs to time gap between two successive trips will have nan values
    # assign a constant value of 'NoActiveTrip' to such data points
    resampled_gps_df.fillna({'TripLogId': 'NoActiveTrip'}, inplace=True)
    logger.debug("Assigned TripLogId to resampled data points")

    return resampled_gps_df