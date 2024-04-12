import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

def get_trip_animation(
    info_df,
    gps_df,
    mapbox_token=None,
    zoom=14,
    height=800,
    transition_time=1000.0
    ):
    # set number of frames in animations
    number_frames = gps_df.shape[0]

    # metre to kilometre conversion factor
    m_to_km = 1/1000
    mps_to_kmph = 3600/1000

    # set axes bounds
    xaxis_range = [gps_df.Timestamp.min(), gps_df.Timestamp.max()]
    yaxis1_range = [min(gps_df.CumulativeDistance.min()*m_to_km, 
                        gps_df.Uncertainty.min()*m_to_km, 0),
                    max(gps_df.CumulativeDistance.max()*m_to_km, 
                        gps_df.Uncertainty.max()*m_to_km)]
    yaxis2_range = [gps_df.Velocity.min()*mps_to_kmph, gps_df.Velocity.max()*mps_to_kmph]

    # specify mapbox layout settings
    mapbox=dict(
        style="satellite" if mapbox_token else 'open-street-map',
        accesstoken=mapbox_token,
        center=dict(
            lat=info_df.iloc[0].LoadLatitude,
            lon=info_df.iloc[0].LoadLongitude),
        zoom=zoom)

    # create a figure with required number of sub plots, plot types, other specs
    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.1,
        x_title="Date & Time",
        specs=[
            [dict(type="mapbox", rowspan=2)],
            [None],
            [dict(type="xy", secondary_y=True)]],
        subplot_titles=(
            "Trip Route",
            "Movement Data"
            )
        )

    # TRACE 0: Trip route as per original gps pings data
    fig.add_trace(
        go.Scattermapbox(
            lat=gps_df.Latitude,
            lon=gps_df.Longitude,
            mode="lines",
            marker=dict(size=5, color='red'),
            name="Trip Route"),
        row=1,
        col=1
        )

    # TRACE 1: Start Point - first gps ping of trip
    fig.add_trace(
        go.Scattermapbox(
            lat=gps_df.iloc[0:1].Latitude,
            lon=gps_df.iloc[0:1].Longitude,
            mode="markers",
            marker=dict(allowoverlap=True, size=6, symbol='bus'),
            showlegend=False,
            name="Current GPS Ping"),
        row=1,
        col=1
        )

    # TRACE 2: End Point - last gps ping of the trip
    fig.add_trace(
        go.Scattermapbox(
            lat=gps_df.iloc[-1:].Latitude,
            lon=gps_df.iloc[-1:].Longitude,
            mode="markers",
            marker=dict(allowoverlap=True, size=10, symbol='embassy'),
            showlegend=False,
            name="End Point"),
        row=1,
        col=1
        )

    # TRACE 3: "Designated Load Point" as per of trip info data
    fig.add_trace(
        go.Scattermapbox(
            lat=info_df.iloc[0:1].LoadLatitude,
            lon=info_df.iloc[0:1].LoadLongitude,
            mode="markers",
            marker=dict(allowoverlap=True, size=8, color='yellow', symbol='circle', opacity=0.8),
            showlegend=True,
            name="Load Point"),
        row=1,
        col=1
        )

    # TRACE 4: "Designated Dump Point" as per of trip info data
    fig.add_trace(
        go.Scattermapbox(
            lat=info_df.iloc[0:1].DumpLatitude,
            lon=info_df.iloc[0:1].DumpLongitude,
            mode="markers",
            marker=dict(allowoverlap=True, size=8, color='cyan', symbol='circle', opacity=0.8),
            showlegend=True,
            name="Dump Point"),
        row=1,
        col=1
        )

    # TRACE 5: Plot cumulative distance covered over time
    fig.add_trace(
        go.Scatter(
            x=gps_df.Timestamp,
            y=gps_df.CumulativeDistance*m_to_km,
            mode="markers",
            marker=dict(color='blue', size=3), 
            opacity=0.5, 
            cliponaxis=False,
            name="Cumulative Distance (km)"),
        secondary_y=False,
        row=3,
        col=1
        )

    # TRACE 6: Plot velocity between successibe gps to get a notion of speed
    fig.add_trace(
        go.Scatter(
            x=gps_df.Timestamp,
            y=gps_df.Velocity*(mps_to_kmph),
            mode="markers",
            marker=dict(color='magenta', size=3), 
            opacity=0.8, 
            cliponaxis=False, 
            name="Speed (kmph)"),
        secondary_y=True,
        row=3,
        col=1
        )

    # create frames with corresponding data that needs to be rendered
    frames = [
        dict(
            name=k,
            data=[
                go.Scattermapbox(visible=True),
                go.Scattermapbox(
                    lat=gps_df.iloc[k:k+1]["Latitude"],
                    lon=gps_df.iloc[k:k+1]["Longitude"]
                    ),
                go.Scattermapbox(visible=True),
                go.Scattermapbox(visible=True),
                go.Scattermapbox(visible=True),
                go.Scatter(
                    x=gps_df.iloc[0:k]["Timestamp"],
                    y=gps_df.iloc[0:k]["CumulativeDistance"]*(m_to_km)
                    ),
                go.Scatter(
                    x=gps_df.iloc[0:k]["Timestamp"],
                    y=gps_df.iloc[0:k]["Velocity"]*(mps_to_kmph)
                    )
                ],
            traces=[0, 1, 2, 3, 4, 5, 6]
            ) for k in range(number_frames)
            ]

    # define behavior of Play and Pause buttons
    updatemenus_list = [
        dict(
            type='buttons',
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[
                        [f'{k}' for k in range(number_frames)],
                        dict(
                            frame=dict(
                                duration=transition_time, redraw=True),
                                mode='immediate',
                                transition=dict(duration=transition_time),
                                easing='linear',
                                fromcurrent=True
                            )
                        ]
                    ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                            )
                        ]
                    )
                ],
             direction= 'left',
             pad=dict(r=10, t=85),
             showactive=True,
             x=0.1,
             y=0,
             xanchor='right',
             yanchor='top'
             )
        ]

    # define behavior of slider control for frames
    sliders_list = [
        dict(
            yanchor='top',
            xanchor='left',
            currentvalue=dict(
                font=dict(size=10),
                prefix='Frame: ',
                visible=True,
                xanchor='right'),
            transition=dict(duration=transition_time, easing='linear'),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[
                dict(
                    args=[[k],
                          dict(
                              frame=dict(
                                  duration=0.0,
                                  easing='linear',
                                  redraw=True
                                  ),
                              transition=dict(duration=0.0, easing='linear')
                              )
                          ],
                    label=k,
                    method='animate'
                    ) for k in range(number_frames)
                    ]
            )
        ]

    fig.update(frames=frames)
    fig.update_layout(
        mapbox=mapbox,
        height=height,
        updatemenus=updatemenus_list,
        sliders=sliders_list
        )
    fig.update_layout(
        title="Animation: Trip Route & Movement Parameters",
        yaxis_title="Cumulative Distance (km)",
        yaxis2_title="Velocity (kmph)",
        updatemenus=updatemenus_list,
        sliders=sliders_list
        )
    fig["layout"]["xaxis"]["range"] = xaxis_range
    fig["layout"]["yaxis"]["range"] = yaxis1_range
    fig["layout"]["yaxis2"]["range"] = yaxis2_range

    return fig