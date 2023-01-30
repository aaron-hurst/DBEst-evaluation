from time import perf_counter
from datetime import datetime
import json
import logging
import os

import pandas as pd
from dbestclient.executor.executor import SqlExecutor

from config import LOG_FORMAT, NAME_DELIMITER, DATA_DIR, WAREHOUSE_DIR, QUERIES_DIR
from schemas import DB_SCHEMAS

LOGGING_LEVEL = logging.INFO
DATA_MODEL_CREATED = False
DATA_SOURCE = "ampds"
DATASET_ID = "basement_plugs_and_lights"
QUERIES_SET = "ampds-basement_plugs_and_lights-N=100"
SAMPLING_METHOD = "uniform"
SAMPLE_SIZE = 1000
SAVE_SAMPLE = True
NUM_EPOCH = 400
DENSITY_TYPE = "kde"
REGRESSION_TYPE = "mdn"
AGGREGATIONS = [
    "COUNT",
    "SUM",
    "AVG",
    "MEDIAN",
    "MAX",
    "MIN",
    "VARIANCE",
]
# TODO check running_parameters.DbestConfig for more parameters


def build_models(dataset_full_id, csv_path):
    """Build/load models for each pair of columns and return SQL executor object."""
    schema = DB_SCHEMAS[DATA_SOURCE][DATASET_ID]
    sql_executor = SqlExecutor(WAREHOUSE_DIR, SAVE_SAMPLE, NUM_EPOCH, DENSITY_TYPE)
    n_columns = len(schema["column_names"])
    for i in range(n_columns):
        column_1_str = schema["column_names"][i] + " " + schema["column_types"][i]
        for j in range(n_columns):
            logger.debug(f"Building model for column pair: ({i}, {j})")
            column_2_str = schema["column_names"][j] + " " + schema["column_types"][j]
            table_name = (
                f"{dataset_full_id}_{schema['column_names'][i]}_"
                f"{schema['column_names'][j]}_{SAMPLE_SIZE}"
            )
            sql_create_model = (
                f"create table "
                f"{table_name}({column_1_str}, {column_2_str}) "
                f"from '{csv_path}' "
                f"method {SAMPLING_METHOD} size {SAMPLE_SIZE};"
            )
            try:
                sql_executor.execute(sql_create_model)
            except FileExistsError:
                logger.info(f"Model {table_name} already exists.")
                continue
    return sql_executor


def main():
    # Setup
    dataset_full_id = DATA_SOURCE + NAME_DELIMITER + DATASET_ID
    schema = DB_SCHEMAS[DATA_SOURCE][DATASET_ID]
    csv_path = os.path.join(DATA_DIR, "uncompressed", dataset_full_id + ".csv")
    queries_path = os.path.join(QUERIES_DIR, dataset_full_id, QUERIES_SET + ".csv")
    results_path = os.path.join(
        "experiments",
        "comparisons_for_paper",
        dataset_full_id,
        QUERIES_SET + "_" + str(SAMPLE_SIZE),
    )
    results_filepath = os.path.join(results_path, "results.csv")
    info_filepath = os.path.join(results_path, "info.txt")

    # Ensure output directories exists
    if os.path.exists(WAREHOUSE_DIR):
        logger.info("Warehouse is initialized.")
    else:
        logger.info("Warehouse does not exists, so initialize one.")
        os.makedirs(WAREHOUSE_DIR)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Create models
    # method <method> <rate> specifies the sampling rate for creating the models.
    # e.g. "method uniform size 1000" does uniform sampling with 1000 samples
    t_modelling_start = perf_counter()
    logger.info("Creating data model...")
    sql_executor = build_models(dataset_full_id, csv_path)
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
        if (i % 100) == 0:
            logger.info(f"Running query {i}")

        if i > 5:
            break

        for aggregation_col in range(n_columns):
            aggregation_col_name = schema["column_names"][aggregation_col]
            model_name = (
                f"{dataset_full_id}_{aggregation_col_name}"
                f"_{predicate_col_name}_{SAMPLE_SIZE}"
            )
            for aggregation in AGGREGATIONS:
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
                results.append(
                    {'query_id': i,
                    "predicate_column": predicate_col,
                    "aggregation_column": aggregation_col,
                    "aggregation": aggregation,
                    'predicted_value': predicted_value,
                    'latency': t_estimate,
                    })
                n_queries = n_queries + 1
    t_queries = perf_counter() - t_queries_start

    # Export results
    df = pd.DataFrame(results).set_index("query_id", drop=True)
    logger.info("Exporting results.")
    df.to_csv(results_filepath, index=True)

    # Get storage information
    s_original = schema["n_rows"] * n_columns * 4  # bytes
    s_models = 0
    for file_name in os.listdir(WAREHOUSE_DIR):
        if file_name.startswith(dataset_full_id) and file_name.endswith(".pkl"):
            s_models += os.stat(os.path.join(WAREHOUSE_DIR, file_name)).st_size

    # Export configuration and performance statistics
    os.makedirs(os.path.dirname(info_filepath), exist_ok=True)
    logger.info(f"Saving experiment parameters and statistics to {info_filepath}")
    with open(info_filepath, "w", newline="") as f:
        f.write(f"------------- Parameters -------------\n")
        f.write(f"DATA_MODEL_CREATED   {DATA_MODEL_CREATED}\n")
        f.write(f"DATA_SOURCE          {DATA_SOURCE}\n")
        f.write(f"DATASET_ID           {DATASET_ID}\n")
        f.write(f"QUERIES_SET          {QUERIES_SET}\n")
        f.write(f"SAMPLING_METHOD      {SAMPLING_METHOD}\n")
        f.write(f"SAMPLE_SIZE          {SAMPLE_SIZE:,d}\n")
        f.write(f"NUM_EPOCH            {NUM_EPOCH:,d}\n")
        f.write(f"DENSITY_TYPE         {DENSITY_TYPE}\n")

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

if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    main()
