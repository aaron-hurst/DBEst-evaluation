from time import perf_counter
from datetime import datetime
import json
import numpy as np
import logging
import os

import pandas as pd
from dbestclient.executor.executor import SqlExecutor

from config import LOG_FORMAT, NAME_DELIMITER, DATA_DIR, WAREHOUSE_DIR, QUERIES_DIR
from schemas import DB_SCHEMAS

LOGGING_LEVEL = logging.INFO
# DATA_SOURCE = "kaggle"
# DATASET_ID = "temperature_iot_on_gcp"
# QUERIES_SET = "kaggle-temperature_iot_on_gcp-N=100"
# SAMPLING_METHOD = "uniform"
# SAMPLE_SIZE = 10000
SAVE_SAMPLE = True
AGGREGATIONS = [
    "COUNT",
    "SUM",
    "AVG",
    "MEDIAN",
    "MAX",
    "MIN",
    "VARIANCE",
]


def load_dataset(dataset_full_id, csv_path):
    if dataset_full_id in ["chicago-taxi_trips_2020", "kaggle-light_detection"]:
        df = pd.read_csv(csv_path, header=0, dtype=np.float64)
        n_bytes_per_value = 8
    else:
        df = pd.read_csv(csv_path, header=0, dtype=np.float32)
        n_bytes_per_value = 4
    return df.dropna(), n_bytes_per_value


def build_models(
    dataset_full_id, csv_path, warehouse_path, sample_size, sampling_method="uniform"
):
    """Build/load models for each pair of columns and return SQL executor object."""
    data_source, dataset_id = dataset_full_id.split(NAME_DELIMITER)
    schema = DB_SCHEMAS[data_source][dataset_id]
    sql_executor = SqlExecutor(warehouse_path, save_sample=SAVE_SAMPLE)
    n_columns = len(schema["column_names"])
    for i in range(n_columns):
        column_1_str = schema["column_names"][i] + " real"
        for j in range(n_columns):
            logger.debug(f"Building model for column pair: ({i}, {j})")
            column_2_str = schema["column_names"][j] + " real"
            table_name = (
                f"{dataset_full_id}_{schema['column_names'][i]}_"
                f"{schema['column_names'][j]}_{sample_size}"
            )
            sql_create_model = (
                f"create table "
                f"{table_name}({column_1_str}, {column_2_str}) "
                f"from '{csv_path}' "
                f"method {sampling_method} size {sample_size};"
            )
            try:
                sql_executor.execute(sql_create_model)
            except FileExistsError:
                logger.info(f"Model {table_name} already exists.")
                continue
    return sql_executor


def aggregate(data, agg):
    if agg.upper() == "COUNT":
        return data.shape[0]
    elif agg.upper() == "SUM":
        return data.sum()
    elif agg.upper() == "AVG":
        return data.mean()
    else:
        raise NotImplementedError("Aggregation {:s} not available".format(agg))


def run_experiment(
    data_source, dataset_id, sample_size, sampling_method="uniform", testing_flag=False
):
    logger.info(
        "Running experiment for:"
        f"\n\tdata source:  {data_source}"
        f"\n\tdataset id:   {dataset_id}"
        f"\n\tsample size:  {sample_size:,d}"
    )

    # Setup
    dataset_full_id = data_source + NAME_DELIMITER + dataset_id
    query_set = dataset_full_id + NAME_DELIMITER + "N=100"
    schema = DB_SCHEMAS[data_source][dataset_id]
    csv_path = os.path.join(DATA_DIR, "uncompressed", dataset_full_id + ".csv")
    warehouse_path = os.path.join(
        WAREHOUSE_DIR, dataset_full_id, f"sample_size_{sample_size}"
    )
    queries_path = os.path.join(QUERIES_DIR, dataset_full_id, query_set + ".csv")
    results_path = os.path.join(
        "experiments",
        "comparisons_for_paper",
        dataset_full_id,
        query_set,
        f"sample_size_{sample_size}",
    )
    if testing_flag:  # append timestamp to results dir if this is just a test run
        results_path = results_path + "_test_" + datetime.now().strftime("%Y%m%d%H%M%S")
    results_filepath = os.path.join(results_path, "results.csv")
    info_filepath = os.path.join(results_path, "info.txt")
    data, n_bytes_per_value = load_dataset(dataset_full_id, csv_path)

    # Ensure output directories exists
    if os.path.exists(warehouse_path):
        logger.info("Warehouse is initialized.")
    else:
        logger.info("Warehouse does not exists, so initialize one.")
        os.makedirs(warehouse_path)

    # Create models
    # method <method> <rate> specifies the sampling rate for creating the models.
    # e.g. "method uniform size 1000" does uniform sampling with 1000 samples
    t_modelling_start = perf_counter()
    logger.info("Creating data model...")
    sql_executor = build_models(dataset_full_id, csv_path, warehouse_path, sample_size)
    t_modelling = perf_counter() - t_modelling_start

    # Run queries
    logger.info("Evaluating queries")
    n_columns = len(schema["column_names"])
    df_queries = pd.read_csv(queries_path, dtype={"predicate_column": int})
    results = []
    n_queries = 0
    t_queries_start = perf_counter()
    for i, query in df_queries.iterrows():
        predicate_col = int(query.predicate_column)
        predicate_col_name = schema["column_names"][predicate_col]
        predicate = [query.predicate_low, query.predicate_high]
        mask = (data[predicate_col_name] > predicate[0]) & (
            data[predicate_col_name] < predicate[1]
        )
        if (i % 50) == 0:
            logger.info(f"Running query {i}")
        for aggregation_col in range(n_columns):
            aggregation_col_name = schema["column_names"][aggregation_col]
            model_name = (
                f"{dataset_full_id}_{aggregation_col_name}"
                f"_{predicate_col_name}_{sample_size}"
            )
            for aggregation in AGGREGATIONS:
                # Predicated value
                query_str = (
                    f"select {aggregation}({aggregation_col_name}) "
                    f"from {model_name} "
                    f"where {predicate_col_name} "
                    f"between {predicate[0]} and {predicate[1]}"
                )
                try:
                    predicted_value, t_estimate = sql_executor.execute(query_str)
                except NotImplementedError:
                    continue

                # Exact value
                exact = aggregate(data.loc[mask][aggregation_col_name], aggregation)

                # Results list
                results.append(
                    {
                        "query_id": i,
                        "predicate_column": predicate_col,
                        "aggregation_column": aggregation_col,
                        "specificity": query.specificity,
                        "aggregation": aggregation,
                        "latency": t_estimate,
                        "predicted_value": predicted_value,
                        "exact_value": exact,
                    }
                )
                n_queries = n_queries + 1
    t_queries = perf_counter() - t_queries_start

    # Compute error statistics
    df = pd.DataFrame(results).set_index("query_id", drop=True)
    df["error"] = df["predicted_value"] - df["exact_value"]
    df["relative_error"] = (
        (df["error"] / df["exact_value"] * 100)
        .abs()
        .replace({np.nan: 100, np.inf: np.nan, -np.inf: np.nan})
        
    )

    # Export results
    logger.info("Exporting results.")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    df.to_csv(results_filepath, index=True)

    # Display results
    logger.info(
        "Median relative error by column:\n%s",
        df.groupby(["predicate_column", "aggregation_column"])
        .agg({"relative_error": "median"})
        .unstack()
        .round(2),
    )
    logger.info(
        "Relative error by accregate:\n%s",
        df.groupby("aggregation")[["relative_error"]]
        .describe(percentiles=[0.5, 0.75, 0.95])
        .round(3),
    )

    # Get storage information
    s_original = schema["n_rows"] * n_columns * n_bytes_per_value
    s_models = 0
    for file_name in os.listdir(warehouse_path):
        if file_name.startswith(dataset_full_id) and file_name.endswith(".pkl"):
            s_models += os.stat(os.path.join(warehouse_path, file_name)).st_size

    # Export configuration and performance statistics
    os.makedirs(os.path.dirname(info_filepath), exist_ok=True)
    logger.info(f"Saving experiment parameters and statistics to {info_filepath}")
    n_epoch = sql_executor.config.get_parameter("num_epoch")
    n_gaussians = sql_executor.config.get_parameter("num_gaussians")
    regression_type = sql_executor.config.get_parameter("reg_type")
    density_type = sql_executor.config.get_parameter("density_type")
    device_type = sql_executor.config.get_parameter("device")
    n_mdn_layer_node = sql_executor.config.get_parameter("n_mdn_layer_node")
    integration_epsabs = sql_executor.config.get_parameter("epsabs")
    integration_epsrel = sql_executor.config.get_parameter("epsrel")
    integration_limit = sql_executor.config.get_parameter("limit")
    with open(info_filepath, "w", newline="") as f:
        f.write(f"------------- Parameters -------------\n")
        f.write(f"DATA_SOURCE          {data_source}\n")
        f.write(f"DATASET_ID           {dataset_id}\n")
        f.write(f"QUERIES_SET          {query_set}\n")
        f.write(f"SAMPLING_METHOD      {sampling_method}\n")
        f.write(f"SAMPLE_SIZE          {sample_size:,d}\n")
        f.write(f"N_EPOCH              {n_epoch:,d}\n")
        f.write(f"N_GAUSSIANS          {n_gaussians:,d}\n")
        f.write(f"N_MDN_LAYER_NODE     {n_mdn_layer_node:,d}\n")
        f.write(f"REGRESSION_TYPE      {regression_type}\n")
        f.write(f"DENSITY_TYPE         {density_type}\n")
        f.write(f"DEVICE_TYPE          {device_type}\n")
        f.write(f"INTEGRATION_EPSABS   {integration_epsabs}\n")
        f.write(f"INTEGRATION_EPSREL   {integration_epsrel}\n")
        f.write(f"INTEGRATION_LIMIT    {integration_limit}\n")

        f.write(f"\n------------- Runtime -------------\n")
        f.write(f"Generate models      {t_modelling:.3f} s\n")
        f.write(f"Run queries          {t_queries:.3f} s\n")
        f.write(f"Queries executed     {n_queries}\n")
        if n_queries:
            mean_latency = t_queries / n_queries
        else:
            mean_latency = 0
        f.write(f"Mean latency         {mean_latency:.6f} s\n")

        f.write(f"\n------------- Storage -------------\n")
        f.write(f"Original data        {s_original:,d} bytes\n")
        f.write(f"Models               {s_models:,d} bytes\n")
        f.write(f"Models (%)           {s_models / s_original * 100:.2f} %\n")
    return


def main():
    # Run all experiments (all datasets and multiple sample sizes)
    # for data_source in DB_SCHEMAS:
    #     for dataset_id in DB_SCHEMAS[data_source]:
    #         for sample_size in [1000]:  # try 10000 later for at least some datasets
    #             run_experiment(data_source, dataset_id, sample_size)

    # Run a single experiment
    # run_experiment("kaggle", "aquaponics_all", 1000)
    # run_experiment("kaggle", "smart_building_system_all", 1000)
    # run_experiment("kaggle", "temperature_iot_on_gcp_100k", 1000)
    # run_experiment("uci", "household_power_consumption", 1000)
    # run_experiment("uci", "gas_sensor_home_activity", 1000)

    run_experiment("kaggle", "light_detection", 1000, testing_flag=True)


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    main()
