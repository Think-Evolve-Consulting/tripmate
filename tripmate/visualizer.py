import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import LineString, MultiLineString

logger = logging.getLogger(__name__)

def get_plottable_overlaps(
    overlapping_trips_df
):
    plottable_overlaps_df = \
        pd.concat(
            [
                overlapping_trips_df\
                    [['DumperMachineNumber', 'TripLogId', 'TripName', 'TripStart', 'TripSequence']]\
                    .rename(columns={'TripStart': 'Timestamp'}), 
                overlapping_trips_df\
                    [['DumperMachineNumber', 'TripLogId', 'TripName', 'TripEnd', 'TripSequence']]\
                    .rename(columns={'TripEnd': 'Timestamp'}), 
                overlapping_trips_df\
                    [['DumperMachineNumber', 'TripLogId', 'TripName', 'TripEnd']]\
                    .rename(columns={'TripEnd': 'Timestamp'})
            ], 
            ignore_index=True
        )\
        .sort_values(
            by=['DumperMachineNumber', 'TripLogId', 'Timestamp', 'TripSequence'], 
            ascending=[True, True, True, False]
            )\
        .reset_index(drop=True)
    
    return plottable_overlaps_df

def get_plottable_segments(
    trips_segments_df
):
    plottable_segments_df = \
        pd.concat(
            [
                trips_segments_df\
                    [['SegmentStart', 'DistanceStart']]\
                    .rename(columns={'SegmentStart': 'Timestamp', 'DistanceStart': 'CumulativeDistance'}), 
                trips_segments_df\
                    [['SegmentEnd', 'DistanceEnd']]\
                    .rename(columns={'SegmentEnd': 'Timestamp', 'DistanceEnd': 'CumulativeDistance'}), 
                trips_segments_df\
                    [['SegmentEnd']]\
                    .rename(columns={'SegmentEnd': 'Timestamp'})
            ], 
            ignore_index=True
        )\
        .sort_values(by=['Timestamp', 'CumulativeDistance'], ascending=[True, False])\
        .reset_index(drop=True)
    
    return plottable_segments_df

def plot_activity_detections(
    trips_gps_subset_df, 
    trips_segments_df
):
    title_text = "Loading / Dumping Activity Detections"
    machine_count = trips_gps_subset_df.DumperMachineNumber.nunique()
    if machine_count == 1:
        machine_number = trips_gps_subset_df.DumperMachineNumber.astype(int).astype(str).unique()[0]
        title_text += f" - Machine Number: {machine_number}"

    # convert trips segments into a format suitable for plotting individual segments
    acceleration_class_df = \
        get_plottable_segments(
            trips_segments_df.query("AccelerationClass == 'IDLE'").copy()
        )
    velocity_class_df = \
        get_plottable_segments(
            trips_segments_df.query("VelocityClass == 'IDLE'").copy()
        )
    on_site_class_df = \
        get_plottable_segments(
            trips_segments_df.query("LocationClass == 'ON-SITE'").copy()
        )
    load_dump_class_df = \
        get_plottable_segments(
            trips_segments_df\
                .query("AccelerationClass == 'IDLE'")\
                .query("VelocityClass == 'IDLE'")\
                .query("LocationClass == 'ON-SITE'")\
                .copy()
        )

    # plot actual gps pings based cumulative distance
    activity_figure = \
        px.line(
            trips_gps_subset_df,
            x="Timestamp",
            y="CumulativeDistance",
            color="TripLogId",
            labels={"CumulativeDistance": "Cumulative Distance (m)"}, 
            )

    # workaround to remove TripLogId values from showing in legends section
    for i in range(trips_gps_subset_df.TripLogId.nunique()):
        activity_figure.data[i]['showlegend'] = False
    
    # workaround to remove TripLogId text showing up as legend group title
    activity_figure.update_layout({'legend_title_text': ''})

    # plot thick color band for segments that belong to load or dump activities
    activity_figure.add_trace(
        go.Scatter(
            x=velocity_class_df.Timestamp,
            y=velocity_class_df.CumulativeDistance,
            mode='lines',
            opacity=0.5,
            line=dict(color='green', width=10),
            name='Loading / Dumping Activity', 
            showlegend=True
        )
    )

    # plot thick color band for segments that have velocity class as IDLE
    activity_figure.add_trace(
        go.Scatter(
            x=velocity_class_df.Timestamp,
            y=velocity_class_df.CumulativeDistance,
            mode='lines',
            opacity=0.25,
            line=dict(color='blue', width=10),
            name='Velocity Class: Idle', 
            showlegend=True
        )
    )

    # plot thick color band for segments that have acceleration class as IDLE
    activity_figure.add_trace(
        go.Scatter(
            x=acceleration_class_df.Timestamp,
            y=acceleration_class_df.CumulativeDistance,
            mode='lines',
            opacity=0.25,
            line=dict(color='magenta', width=10),
            name='Acceleration Class: Idle', 
            showlegend=True
        )
    )

    # plot thick color band for segments that have location class as ON SITE
    activity_figure.add_trace(
        go.Scatter(
            x=on_site_class_df.Timestamp,
            y=on_site_class_df.CumulativeDistance,
            mode='lines',
            opacity=0.25,
            line=dict(color='gold', width=15),
            name='Location Class: On Site', 
            showlegend=True
        )
    )

    activity_figure.update_layout(
        title=title_text
    )

    return activity_figure

def plot_simplified_routes(
    routes_gdf, 
    color_column='DumperMachineNumber', 
    mapbox_style='open-street-map', 
    mapbox_token=None, 
    zoom=14, 
    height=600, 
    opacity=0.75
    ):
    # get date range of given data
    min_date = routes_gdf.TripDateString.min()
    max_date = routes_gdf.TripDateString.max()
    trips_count = routes_gdf.TripLogId.nunique()
    title_text = \
        f"Routes of {trips_count} trips between {min_date} and {max_date}"

    # convert shapefile data into plotly readable format
    linestrings = x = y = lats = lons = trips = machines = None

    for feature, trip, machine in \
        zip(
            routes_gdf.geometry, 
            routes_gdf.TripName, 
            routes_gdf.DumperMachineNumber):
        # get line strings from all routes
        if isinstance(feature, LineString):
            linestrings = [feature]
        elif isinstance(feature, MultiLineString):
            linestrings = feature.geoms
        else:
            continue

        # process each route's linstring one by one
        for linestring in linestrings:
            # get the longitude and latitude values
            x, y = linestring.xy
            # save all coordinates in the sequence given
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            trips = np.append(trips, [trip]*len(y))
            machines = np.append(machines, [str(int(machine))]*len(y))
            # add None to prevent joining together different routes
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            trips = np.append(trips, None)
            machines = np.append(machines, None)
    
    # create a pandas data frame from lats, lons, trips and machines arrays
    routes_df = \
        pd.DataFrame(
            dict(
                Latitude=lats, 
                Longitude=lons, 
                TripName=trips, 
                DumperMachineNumber=machines
                )
            )

    # specify mapbox layout settings
    mapbox=dict(
        style=mapbox_style,
        accesstoken=mapbox_token,
        center=\
            dict(
                lat=routes_df.iloc[1].Latitude,
                lon=routes_df.iloc[1].Longitude
                ),
        zoom=zoom
        )

    # plot all the lines from derived routes data frame
    routes_figure = \
        px.line_mapbox(
            data_frame=routes_df, 
            lat='Latitude', 
            lon='Longitude', 
            color=color_column,  
            hover_data='DumperMachineNumber' if color_column == 'TripName' else 'TripName', 
            title=title_text
        )
    
    # workaround to set line width and opacity
    for element in routes_figure.data:
        element.line['width'] = 3
        element['opacity'] = 0.75

    # update figure layout parameters
    routes_figure.update_layout(
        mapbox=mapbox, 
        legend_title=color_column
        )

    return routes_figure

def plot_routes(
    gps_df, 
    color_column='DumperMachineNumber', 
    mapbox_style='open-street-map', 
    mapbox_token=None, 
    zoom=14, 
    height=600, 
    opacity=0.75
    ):
    # get date range of given data
    min_date = gps_df.TripDateString.min()
    max_date = gps_df.TripDateString.max()
    trips_count = gps_df.TripLogId.nunique()
    title_text = \
        f"Routes of {trips_count} trips between {min_date} and {max_date}"

    # change DumperMachineNumber to string
    gps_df['DumperMachineNumber'] = gps_df.DumperMachineNumber.astype(int).astype(str)

    # specify mapbox layout settings
    mapbox=dict(
        style=mapbox_style,
        accesstoken=mapbox_token,
        center=\
            dict(
                lat=gps_df.iloc[0].Latitude,
                lon=gps_df.iloc[0].Longitude
                ),
        zoom=zoom
        )

    # plot all the routes from given gps data frame
    routes_figure = \
        px.scatter_mapbox(
            data_frame=gps_df, 
            lat='Latitude', 
            lon='Longitude', 
            color=color_column, 
            hover_name='TripName', 
            opacity=opacity, 
            title=title_text, 
            height=height
            )

    # update figure layout parameters
    routes_figure.update_layout(
        mapbox=mapbox, 
        legend_title=color_column
        )

    return routes_figure

def plot_machine_routes(
    machine_gps_df, 
    mapbox_style='open-street-map', 
    mapbox_token=None, 
    zoom=14, 
    height=600, 
    opacity=0.75
    ):
    # get machine number from gps data
    machine_number = machine_gps_df.iloc[0].DumperMachineNumber

    # specify mapbox layout settings
    mapbox=dict(
        style=mapbox_style,
        accesstoken=mapbox_token,
        center=\
            dict(
                lat=machine_gps_df.iloc[0].Latitude,
                lon=machine_gps_df.iloc[0].Longitude
                ),
        zoom=zoom
        )

    # plot all the routes from given gps data frame
    machine_routes_figure = \
        px.scatter_mapbox(
            data_frame=machine_gps_df, 
            lat='Latitude', 
            lon='Longitude', 
            color='TripName', 
            opacity=opacity, 
            title=f"Trip Routes for Machine Number {int(machine_number)}", 
            height=height
            )

    # update figure layout parameters
    machine_routes_figure.update_layout(
        mapbox=mapbox, 
        legend_title='Trip Name'
        )

    return machine_routes_figure

def plot_clustering_results_2d(
    movement_params_df, 
    pca_loadings_df, 
    clustering_method, 
    height=600, 
    width=1200
    ):
    # choose cluster label column based on provided method
    cluster_label = f"TSNE{clustering_method}Cluster"

    # create a figure with required number of sub plots, plot types, other specs
    fig = make_subplots(
        rows=1,
        cols=2, 
        horizontal_spacing=0.1, 
        specs=[
            [dict(type="xy", secondary_y=False),
            dict(type="xy", secondary_y=False)]],
        subplot_titles=(
            "Clusters in t-SNE subspace",
            "Clusters in PCA subspace with loadings"
            )
        )

    # TRACE 0: Clusters in t-SNE subspace
    fig.add_trace(
        go.Scatter(
            x=movement_params_df.TSNE1,
            y=movement_params_df.TSNE2, 
            hovertext=movement_params_df[cluster_label], 
            mode="markers",
            marker=dict(colorscale='rainbow', color=movement_params_df[cluster_label], symbol='circle-open'), 
            opacity=0.65, 
            name=f"{clustering_method} in t-SNE subspace", 
            showlegend=False
            ),
        row=1,
        col=1
        )

    # scale the x and y column values to be between -1 and 1
    x_data = movement_params_df.PC1.values
    y_data = movement_params_df.PC2.values
    x_data = x_data / (x_data.max() - x_data.min())
    y_data = y_data / (y_data.max() - y_data.min())

    # TRACE 1: Clusters in PCA subspace
    fig.add_trace(
        go.Scatter(
            x=x_data, 
            y=y_data, 
            hovertext=movement_params_df[cluster_label], 
            mode="markers",
            marker=dict(colorscale='rainbow', color=movement_params_df[cluster_label], symbol='circle-open'), 
            opacity=0.65, 
            name=f"{clustering_method} in PCA", 
            showlegend=False
            ),
        row=1,
        col=2
        )

    for feature in pca_loadings_df.index:
        fig.add_trace(
            go.Scatter(
                x=[0, pca_loadings_df.loc[feature, 'PC1'].squeeze()], 
                y=[0, pca_loadings_df.loc[feature, 'PC2'].squeeze()], 
                hovertext=feature, 
                marker= dict(size=10, symbol= "arrow-up", angleref="previous"), 
                opacity=0.6, 
                name=feature
                ), 
            row=1, 
            col=2
        )

    fig.update_layout(
        title=\
            dict(
                text=f"{clustering_method} Clustering on t-SNE embeddings", 
                x=0.5
            ), 
        legend_font_size=9, 
        height=height, 
        width=width
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(title_text=f"t-SNE: Dimension 1", row=1, col=1)
    fig.update_yaxes(title_text=f"t-SNE: Dimension 2", row=1, col=1)
    fig.update_xaxes(title_text=f"Principal Component 1", row=1, col=2)
    fig.update_yaxes(title_text=f"Principal Component 2", row=1, col=2)

    return fig