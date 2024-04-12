import logging
import numpy as np
import pandas as pd
import ruptures as rpt
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

logger = logging.getLogger(__name__)

def build_trips_chain(
    resampled_gps_df    
):
    trips_chain_df = \
        resampled_gps_df\
            .query("TripLogId != 'NoActiveTrip'")\
            .groupby(['TripLogId'])\
            .agg(TripStart=('Timestamp', 'min'),
                TripEnd=('Timestamp', 'max'))\
            .sort_values(by=['TripStart'], ascending=True)\
            .reset_index()\
            .rename(columns={'TripLogId': 'CurrentTrip'})\
            .copy()
    
    trips_chain_df['NextTrip'] = trips_chain_df.CurrentTrip.shift(-1).fillna('NoNextTrip')
    trips_chain_df['NextTripStart'] = trips_chain_df.TripStart.shift(-1)
    trips_chain_df['NextTripEnd'] = trips_chain_df.TripEnd.shift(-1)

    return trips_chain_df

def detect_change_points(
    data, 
    metric='Velocity', 
    kernel='linear', 
    min_size=15, 
    penalty=25, 
    plot=False
):
    algo_kernel = rpt.KernelCPD(kernel=kernel, min_size=min_size)
    algo_kernel.fit(data[metric].values)
    change_points = algo_kernel.predict(pen=penalty)

    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
        ax.plot(data['CumulativeDistance'])
        for change_point in change_points:
            ax.axvline(x=change_point, color='k', linestyle="--")
        ax.set_title(f"KernelCPD algorithm with {kernel.upper()} kernel")
        plt.show()

    return change_points

def get_trips_segments(
    trips_chain_df,
    resampled_gps_df, 
    metric='Velocity', 
    kernel='linear', 
    min_size=15, 
    penalty=25, 
    plot=False
):
    trips_segments_df = pd.DataFrame({})
    first_segment_start = trips_chain_df.iloc[0].TripStart.to_datetime64()
    last_trip = None

    for index, row in trips_chain_df.iterrows():
        # create a chain of two consecutive trips and break it into segments
        # handles operational errors in ending current trip and starting next trip

        # if current trip is same as last successfully processed trip, skip the row
        if row.CurrentTrip == last_trip:
            pass
        else:
            # create a chain of two consecutive trips and break it into segments
            # if no segments have been found yet, create chain from very first date
            if trips_segments_df.shape[0] == 0:
                current_chain_start = first_segment_start
            else:
                # if segments have bee found previously and stored in trips segments,
                # set chain start date based on last segment from previous chain
                current_chain_start = trips_segments_df.iloc[-1].SegmentStart.to_datetime64()
                trips_segments_df.drop(trips_segments_df.tail(1).index, inplace=True)

            # set chain end date based on availability of next trip
            if row.NextTripEnd is not pd.NaT:
                current_chain_end = row.NextTripEnd
            else:
                current_chain_end = row.TripEnd.to_datetime64()

            current_chain_data = \
                resampled_gps_df.loc[
                    resampled_gps_df.Timestamp.between(
                        current_chain_start,
                        current_chain_end
                    )
                ].copy().reset_index(drop=True)

            # detect change points for current chain
            change_points = \
                detect_change_points(
                    data=current_chain_data, 
                    metric=metric, 
                    kernel=kernel, 
                    min_size=min_size, 
                    penalty=penalty, 
                    plot=plot
                    )

            # convert the change points into segments with start and end timestamps
            if len(change_points) > 0:
                # add 0 as the first change point if it does not exist
                if change_points[0] != 0:
                    change_points = [0] + change_points

                # add number of data points as last change point if it does not exist
                if change_points[-1] != current_chain_data.shape[0]:
                    change_points[-1] = current_chain_data.shape[0]

                # create a data frame with segment start and end timestamps
                current_segments_df = pd.DataFrame(dict(StartIndex=change_points))
                current_segments_df['EndIndex'] = current_segments_df.StartIndex.shift(-1)-1
                current_segments_df.dropna(inplace=True)
                current_segments_df['EndIndex'] = current_segments_df.EndIndex.astype(int)
                current_segments_df['DumperMachineNumber'] = \
                    current_chain_data.loc[current_segments_df.EndIndex, 'DumperMachineNumber'].values
                current_segments_df['SegmentStart'] = \
                    current_chain_data.loc[current_segments_df.StartIndex, 'Timestamp'].values
                current_segments_df['SegmentEnd'] = \
                    current_chain_data.loc[current_segments_df.EndIndex, 'Timestamp'].values
                current_segments_df['DistanceStart'] = \
                    current_chain_data.loc[current_segments_df.StartIndex, 'CumulativeDistance'].values
                current_segments_df['DistanceEnd'] = \
                    current_chain_data.loc[current_segments_df.EndIndex, 'CumulativeDistance'].values

                # collate segments of current chain in trips segments data frame
                if current_segments_df.shape[0] > 0:
                    # remove the columns that are not relevant for future use
                    current_segments_df.drop(columns=['StartIndex', 'EndIndex'], inplace=True)
                    # concatenate with trips segments data frame
                    trips_segments_df = \
                        pd.concat(
                            [trips_segments_df, current_segments_df],
                            ignore_index=True
                            )
                    # store the last processed trip log id so that it can be skipped
                    if row.NextTripEnd is not pd.NaT:
                        last_trip = row.NextTrip
                    else:
                        last_trip = row.CurrentTrip

        # reset current chain variables as their scope is only within the for loop
        current_chain_start = None
        current_chain_end = None
        current_chain_data = None
        change_points = None
        current_segments_df = None

    return trips_segments_df

def classify_by_acceleration(
    trips_segments_df,
    resampled_gps_df,
    acceleration_threshold=0.1
):
    for index, row in trips_segments_df.iterrows():
        # compute mean of absolute acceleration values for the selected segment
        acceleration_mean = \
            np.mean(
                np.abs(
                    resampled_gps_df.loc[
                        resampled_gps_df.Timestamp.between(
                            row.SegmentStart,
                            row.SegmentEnd
                            )
                        ].Acceleration
                    )
                )

        # assign a class based on given threshold
        if round(acceleration_mean, 1) < acceleration_threshold:
            acceleration_class = 'IDLE'
        else:
            acceleration_class = 'MOBILE'

        trips_segments_df.loc[index, 'AccelerationMean'] = acceleration_mean
        trips_segments_df.loc[index, 'AccelerationClass'] = acceleration_class

    return trips_segments_df

def classify_by_velocity(
    trips_segments_df,
    resampled_gps_df,
    velocity_threshold=1
):
    for index, row in trips_segments_df.iterrows():
        # compute mean of velocity values for the selected segment
        velocity_mean = \
            np.mean(
                resampled_gps_df.loc[
                    resampled_gps_df.Timestamp.between(
                        row.SegmentStart,
                        row.SegmentEnd
                        )
                    ].Velocity
                )

        # assign a class based on given threshold
        if round(velocity_mean, 1) < velocity_threshold:
            velocity_class = 'IDLE'
        else:
            velocity_class = 'MOBILE'

        trips_segments_df.loc[index, 'VelocityMean'] = velocity_mean
        trips_segments_df.loc[index, 'VelocityClass'] = velocity_class

    return trips_segments_df

def classify_by_location(
    trips_segments_df,
    trips_gps_df,
    construction_sites_gdf
):
    for index, row in trips_segments_df.iterrows():
        # get gps pings based on segment start and end timestamps
        # subtract 1 second from segment start to handle resampling at round seconds
        gps_pings_array = \
            trips_gps_df.loc[
                ((trips_gps_df.DumperMachineNumber == row.DumperMachineNumber) & \
                (trips_gps_df.Timestamp.dt.tz_localize(None).between(
                    (row.SegmentStart - pd.Timedelta(seconds=1)).to_datetime64(),
                    (row.SegmentEnd + pd.Timedelta(seconds=0)).to_datetime64()
                    )))
                ][['Longitude', 'Latitude']].values

        # check if all gps pings are located within any one construction site
        if np.all(
            [np.any(
                Point(lon_lat).within(construction_sites_gdf.geometry)
                ) for lon_lat in gps_pings_array]
            ):
            location_class = 'ON-SITE'
        else:
            location_class = 'OFF-SITE'

        trips_segments_df.loc[index, 'LocationClass'] = location_class
    
    return trips_segments_df

def annotate_activity_classes(
    resampled_gps_df, 
    trips_segments_df
):
    for index, row in trips_segments_df.iterrows():
        # assign acceleration class into resampled gps data frame
        resampled_gps_df.loc[
            resampled_gps_df.Timestamp.between(
                row.SegmentStart,
                row.SegmentEnd
                ),
            'AccelerationClass'
            ] = row.AccelerationClass

        # assign velocity class into resampled gps data frame
        resampled_gps_df.loc[
            resampled_gps_df.Timestamp.between(
                row.SegmentStart,
                row.SegmentEnd
                ),
            'VelocityClass'
            ] = row.VelocityClass

        # assign location class into resampled gps data frame
        resampled_gps_df.loc[
            resampled_gps_df.Timestamp.between(
                row.SegmentStart,
                row.SegmentEnd
                ),
            'LocationClass'
            ] = row.LocationClass

        # assign final activity class into resampled gps data frame
        resampled_gps_df.loc[
            resampled_gps_df.Timestamp.between(
                row.SegmentStart,
                row.SegmentEnd
                ),
            'ActivityClass'
            ] = row.ActivityClass
    
    return resampled_gps_df

def classify_segments(
    trips_gps_df, 
    resampled_gps_df, 
    trips_segments_df, 
    sites_gdf, 
    velocity_threshold=1, 
    acceleration_threshold=0.1
):
    # classify trip segments as per average velocity in each segment
    trips_segments_df = \
        classify_by_velocity(
            trips_segments_df=trips_segments_df.copy(),
            resampled_gps_df=resampled_gps_df.copy(),
            velocity_threshold=velocity_threshold
        )

    # classify trip segments as per average acceleration in each segment
    trips_segments_df = \
        classify_by_acceleration(
            trips_segments_df=trips_segments_df.copy(),
            resampled_gps_df=resampled_gps_df.copy(),
            acceleration_threshold=acceleration_threshold
        )
    
    # classify trip segments as per the location of machine in those segments
    trips_segments_df = \
        classify_by_location(
            trips_segments_df=trips_segments_df.copy(),
            trips_gps_df=trips_gps_df.copy(),
            construction_sites_gdf=sites_gdf
        )

    # final classification of activity based on velocity, acceleration, location classes
    trips_segments_df['ActivityClass'] = 'DRIVE'
    trips_segments_df.loc[
        (trips_segments_df['VelocityClass']=='IDLE') & \
        (trips_segments_df['AccelerationClass']=='IDLE') & \
        (trips_segments_df['LocationClass']=='ON-SITE'), 
        'ActivityClass'
        ] = 'LOAD-DUMP'
    
    # annotate each sample with classifications obtained above
    resampled_gps_df = \
        annotate_activity_classes(
            resampled_gps_df=resampled_gps_df.copy(),
            trips_segments_df=trips_segments_df.copy()
            )
    
    return resampled_gps_df, trips_segments_df