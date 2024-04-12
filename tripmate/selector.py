import logging
import pandas as pd
import ipywidgets as widgets

logger = logging.getLogger(__name__)

def get_machine_selector(
    data_df
    ):
    machine_dropdown = \
        widgets.Dropdown(
            options=data_df.DumperMachineNumber.sort_values().unique(),
            description="Machine"
        )

    return machine_dropdown

def get_date_selector(
    data_df
    ):
    date_dropdown = \
        widgets.Dropdown(
            options=data_df.TripDateString.sort_values().unique(),
            description="Trip Date"
        )
    
    return date_dropdown

def get_date_selectors(
    min_date, 
    max_date
    ):
    date_options = pd.date_range(min_date, max_date, freq='d').strftime("%Y-%m-%d").values

    start_date_dropdown = \
        widgets.Dropdown(
            options=date_options,
            description="Start Date"
        )

    end_date_dropdown = \
        widgets.Dropdown(
            options=date_options,
            description="End Date"
        )

    return start_date_dropdown, end_date_dropdown

def get_trip_selectors(
    trips_info_df
    ):
    machine_dropdown = \
        widgets.Dropdown(
            options=trips_info_df.DumperMachineNumber.sort_values().unique(),
            description="Machine"
        )

    trip_date_dropdown = \
        widgets.Dropdown(
            options=\
                trips_info_df\
                    .query(f"DumperMachineNumber == {machine_dropdown.value}")\
                    .TripDateString.sort_values().unique(),
            description="Trip Date"
        )

    trip_name_dropdown = \
        widgets.Dropdown(
            options=\
                trips_info_df\
                    .query(f"DumperMachineNumber == {machine_dropdown.value}")\
                    .query(f"TripDateString == '{trip_date_dropdown.value}'")\
                    .TripName.unique(),
            description="Trip Name"
            )

    def on_change_machine(machine, trip_date, trip_name, trips_info_df, widget):
        if widget['new'] != widget['old']:
            trip_date.options = \
                trips_info_df\
                    .query(f"DumperMachineNumber == {machine.value}")\
                    .TripDateString.sort_values().unique()
            trip_name.options = \
                trips_info_df\
                    .query(f"DumperMachineNumber == {machine.value}")\
                    .query(f"TripDateString == '{trip_date.value}'")\
                    .TripName.unique()

    def on_change_trip_date(machine, trip_date, trip_name, trips_info_df, widget):
        if widget['new'] != widget['old']:
            trip_name.options = \
                trips_info_df\
                    .query(f"DumperMachineNumber == {machine.value}")\
                    .query(f"TripDateString == '{trip_date.value}'")\
                    .TripName.unique()

    def on_change_trip_name(widget):
        pass

    return (
        machine_dropdown, 
        trip_date_dropdown, 
        trip_name_dropdown, 
        on_change_machine, 
        on_change_trip_date, 
        on_change_trip_name
        )

def get_machine_date_selectors(
    trips_info_df
    ):
    machine_dropdown = \
        widgets.Dropdown(
            options=trips_info_df.DumperMachineNumber.sort_values().unique(),
            description="Machine"
        )

    trip_date_dropdown = \
        widgets.Dropdown(
            options=\
                trips_info_df\
                    .query(f"DumperMachineNumber == {machine_dropdown.value}")\
                    .TripDateString.sort_values().unique(),
            description="Trip Date"
        )

    def on_change_machine(machine, trip_date, trips_info_df, widget):
        if widget['new'] != widget['old']:
            trip_date.options = \
                trips_info_df\
                    .query(f"DumperMachineNumber == {machine.value}")\
                    .TripDateString.sort_values().unique()

    def on_change_trip_date(widget):
        pass

    def on_change_trip_name(widget):
        pass

    return (
        machine_dropdown, 
        trip_date_dropdown, 
        on_change_machine, 
        on_change_trip_date
        )