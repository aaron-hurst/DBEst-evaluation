DB_SCHEMAS = {
    "ampds": {
        "basement_plugs_and_lights": {
            "column_names": [
                "V",
                "I",
                "f",
                "DPF",
                "APF",
                "P",
                "Pt",
                "Q",
                "Qt",
                "S",
                "St",
            ],
            "n_rows": 1051200,
        },
        "current": {
            "column_names": [
                "WHE",
                "RSE",
                "GRE",
                "MHE",
                "B1E",
                "BME",
                "CWE",
                "DWE",
                "EQE",
                "FRE",
                "HPE",
                "OFE",
                "UTE",
                "WOE",
                "B2E",
                "CDE",
                "DNE",
                "EBE",
                "FGE",
                "HTE",
                "OUE",
                "TVE",
                "UNE",
            ],
            "n_rows": 1051200,
        },
        "furnace_and_thermostat": {
            "column_names": [
                "V",
                "I",
                "f",
                "DPF",
                "APF",
                "P",
                "Pt",
                "Q",
                "Qt",
                "S",
                "St",
            ],
            "n_rows": 1051200,
        },
    },
    "chicago": {
        "taxi_trips_2020": {
            "column_names": [
                "trip_seconds",
                "trip_miles",
                "fare",
                "tips",
                "extras",
                "trip_total",
                "pickup_centroid_latitude",
                "pickup_centroid_longitude",
                "dropoff_centroid_latitude",
                "dropoff_centroid_longitude",
            ],
            "n_rows": 3466498,
        },
    },
    "kaggle": {
        "aquapolics_all": {
            "column_names": [
                "entry_id",
                "temperature (C)",
                "turbidity (NTU)",
                "dissolved oxygen (g/ml)",
                "ph",
                "nitrate (g/ml)",
                "fish length (cm)",
                "fish weight (g)",
                "pond_id",
            ],
            "n_rows": 1114796,
        },
        "light_detection": {
            "column_names": [
                "device",
                "co",
                "humidity",
                "light",
                "lpg",
                "motion",
                "smoke",
                "temp",
            ],
            "n_rows": 405184,
        },
        "smart_building_system_413": {
            "column_names": [
                "timestamp",
                "co2",
                "humidity",
                "light",
                "temperature",
                "pir",
            ],
            "n_rows": 130851,
        },
        "smart_building_system_621A": {
            "column_names": [
                "timestamp",
                "co2",
                "humidity",
                "light",
                "temperature",
                "pir",
            ],
            "n_rows": 130874,
        },
        "smart_building_system_all": {
            "column_names": [
                "timestamp",
                "co2",
                "humidity",
                "light",
                "temperature",
                "pir",
                "room_number",
            ],
            "n_rows": 6571465,
        },
        "temperature_iot_on_gcp": {
            "column_names": [
                "timestamp_epoch",
                "temp_c",
                "device_id",
            ],
            "n_rows": 10553597,
        },
        "temperature_iot_on_gcp_100k": {
            "column_names": [
                "timestamp_epoch",
                "temp_c",
                "device_id",
            ],
            "n_rows": 10553597,
        },
        "temperature_iot_on_gcp_500k": {
            "column_names": [
                "timestamp_epoch",
                "temp_c",
                "device_id",
            ],
            "n_rows": 10553597,
        },
    },
    "uci": {
        "household_power_consumption": {
            "column_names": [
                "global_active_power",
                "global_reactive_power",
                "voltage",
                "global_intensity",
                "sub_metering_1",
                "sub_metering_2",
                "sub_metering_3",
            ],
            "n_rows": 2049280,
        },
        "gas_sensor_home_activity": {
            "column_names": [
                "time",
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
                "R7",
                "R8",
                "Temp.",
                "Humidity",
            ],
            "n_rows": 928991,
        },
    },
}
