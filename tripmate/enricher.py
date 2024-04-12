import logging
import pandas as pd

logger = logging.getLogger(__name__)

def enrich_gps_data(
    trips_info_df, 
    trips_gps_df
    ):
    logger.info("Enriching trips info and gps data ...")

    # sort trips info data as per TripLogId
    trips_info_df.sort_values(
        by=["TripLogId"],
        ascending=True,
        ignore_index=True,
        inplace=True
        )
    logger.debug("Sorted trips info data by TripLogId")

    # add DumperMachineNumber from trips info to trips gps data frame
    trips_gps_df = \
        trips_gps_df.merge(
            trips_info_df[["TripLogId", "DumperMachineNumber"]],
            how="left",
            on="TripLogId",
            indicator=False
            )
    logger.debug("Added DumperMachineNumber from trips info to trips gps data")

    # sort trips gps data in ascending order of DumperMachineNumber followed by Timestamp
    trips_gps_df.sort_values(
        ["DumperMachineNumber", "Timestamp"],
        ascending=True,
        ignore_index=True,
        inplace=True
        )
    logger.debug("Sorted trips gps data by DumperMachineNumber and Timestamp")

    # add trip start and trip end timestamps based on gps pings data
    trips_info_df = \
        pd.merge(
            trips_info_df, 
            trips_gps_df\
                .sort_values(by=["TripLogId", "Timestamp"], ascending=True)\
                .groupby(["TripLogId"])\
                .agg(
                    TripStart=("Timestamp", "min"), 
                    TripEnd=("Timestamp", "max"), 
                    ), 
            how="left", 
            on=["TripLogId"]
            )
    trips_info_df["TripDuration"] = (trips_info_df.TripEnd - trips_info_df.TripStart).dt.total_seconds()
    logger.debug("Added trip start, end and duration to trips info data based on first and last gps pings for each trip")

    # add trip date string in YYYY-MM-DD format based on the trip's first gps ping timestamp
    trips_info_df["TripDateString"] = trips_info_df.TripStart.dt.strftime("%Y-%m-%d")
    logger.debug("Added trip date string with format YYYY-mm-dd to trips info data based on trip start date")

    # add a unique trip name of format M#{nnn}_D#{YYYY-MM-DD}_T{nnnn}
    trips_info_df['TripSequence'] = \
        trips_info_df\
            .sort_values(by=['DumperMachineNumber', 'TripDateString', 'TripStart'])\
            .groupby(['DumperMachineNumber', 'TripDateString'])\
            .cumcount()+1

    trips_info_df['TripName'] = \
        'M:' + trips_info_df['DumperMachineNumber'].astype(int).astype(str) + '-' + \
        'D:' + trips_info_df['TripDateString'] + '-' + \
        'T:' + trips_info_df['TripSequence'].astype(str)

    trips_info_df.drop(columns=['TripSequence'], inplace=True)

    # add TripName from trips info to trips gps data frame
    trips_gps_df = \
        trips_gps_df.merge(
            trips_info_df[["TripLogId", "TripName", "TripDateString"]],
            how="left",
            on="TripLogId",
            indicator=False
            )
    logger.debug("Added unique trip name to trips info and gps data using machine number, trip start date")

    logger.info("Enriched trips info and gps data successfully")

    return trips_info_df, trips_gps_df

def enrich_simplified_routes(
    routes_gdf, 
    trips_info_df, 
    new_crs='EPSG:4326'
    ):
    # add DumperMachineNumber, TripName, TripDateString details from trips info data
    routes_gdf = \
        routes_gdf.merge(
            trips_info_df[["TripLogId", "DumperMachineNumber", "TripName", "TripDateString"]],
            how="left",
            on="TripLogId",
            indicator=False
            )
    logger.debug("Added dumper machine number, trip name and date string from trips info data")

    # convert routes geo data frame to requested coordinate reference system (crs)
    routes_gdf = routes_gdf.to_crs(new_crs)
    logger.debug(f"Converted routes geo data frame to CRS: {new_crs}")

    return routes_gdf