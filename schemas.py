DB_SCHEMAS = {
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
            "column_types": ["real", "real", "real", "real", "real", "real", "real"],
            "n_rows": 2049280,
        }
    },
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
            "column_types": [
                "real",
                "real",
                "real",
                "real",
                "real",
                "real",
                "real",
                "real",
                "real",
                "real",
                "real",
            ],
            "n_rows": 1051200,
        }
    },
}
