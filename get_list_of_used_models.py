import os

from config import QUERIES_DIR

DATASETS = {
    # "ampds-basement_plugs_and_lights": 2,
    # "ampds-current": 2,
    # "ampds-furnace_and_thermostat": 2,
    # "chicago-taxi_trips_2020": 4,
    # "kaggle-aquaponics": 4,
    # "kaggle-light_detection": 2,
    # "kaggle-smart_building_system": 2,
    # "kaggle-temperature_iot_on_gcp": 4,
    # "uci-gas_sensor_home_activity": 2,
    # "uci-household_power_consumption": 2,
    # "usdot-flights": 2,
    # "uci-household_power_consumption": 15,
    # "uci-household_power_consumption_synthetic": 15,
    # "uci-household_power_consumption_10m": 15,
    # "uci-household_power_consumption_100m": 15,
    # "uci-household_power_consumption_1b": 15,
    # "usdot-flights": 4,
    # "usdot-flights_synthetic": 4,
    # "usdot-flights_10m": 4,
    # "usdot-flights_100m": 4,
    # "usdot-flights_1b": 4,
}

for dataset_id, query_set in DATASETS.items():
    print(f"Checking {dataset_id} set {query_set} for required models...")
    queries_filepath = os.path.join(QUERIES_DIR, f"{dataset_id}_v{query_set}.txt")
    with open(queries_filepath) as f:
        queries = f.readlines()
    out_filename = os.path.join(
        "evaluation", f"models_required_{dataset_id}_v{query_set}.txt"
    )
    models = []
    for i, query in enumerate(queries):
        query_split = query.split(" ")
        aggregation_column = query_split[1].split("(")[1].split(")")[0]
        predicate_columns = [
            query_split[i + 1]
            for i in range(len(query_split))
            if query_split[i].lower() in ["where", "and", "or"]
        ]
        total_columns = len(set([aggregation_column, *predicate_columns]))
        if total_columns > 2:
            continue
        models.append(f"{aggregation_column},{list(set(predicate_columns))[0]}")

    # Export model names
    models = sorted(list(set(models)))
    with open(out_filename, "w") as fp:
        for model in models:
            fp.write(f"{model}\n")
