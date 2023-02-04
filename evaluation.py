from time import perf_counter
from datetime import datetime
import numpy as np
import json
import logging
import os

import pandas as pd
from dbestclient.executor.executor import SqlExecutor

from config import LOG_FORMAT, NAME_DELIMITER, DATA_DIR, QUERIES_DIR

LOGGING_LEVEL = logging.INFO
SAVE_SAMPLE = True
AGGREGATIONS = [
    "COUNT",
    "SUM",
    "AVG",
    "MEDIAN",
    "MAX",
    "MIN",
    "VAR",
]
with open("schemas.json", "r") as fp:
    DB_SCHEMAS = json.load(fp)


def load_dataset(csv_path, dataset_full_id=None):
    if dataset_full_id in ["chicago-taxi_trips_2020", "kaggle-light_detection"]:
        df = pd.read_csv(csv_path, header=0, dtype=np.float64)
        n_bytes_per_value = 8
    else:
        df = pd.read_csv(csv_path, header=0, dtype=np.float32)
        n_bytes_per_value = 4
    return df.dropna(), n_bytes_per_value


def build_models(
    dataset_full_id, csv_path, models_dir, sample_size, sampling_method="uniform"
):
    """Build/load models for each pair of columns and return SQL executor object."""
    data_source, dataset_id = dataset_full_id.split(NAME_DELIMITER)
    schema = DB_SCHEMAS[data_source][dataset_id]
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    sql_executor = SqlExecutor(models_dir, save_sample=SAVE_SAMPLE)
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
    elif agg == "VAR":
        return data.var(ddof=0)
    elif agg == "MEDIAN":
        return np.median(data)
    elif agg == "MIN":
        return data.min()
    elif agg == "MAX":
        return data.max()
    elif agg == "PERCENTILE10":
        return np.percentile(data, 10)
    elif agg == "PERCENTILE90":
        return np.percentile(data, 90)
    else:
        raise NotImplementedError("Aggregation {:s} not available".format(agg))


def get_relative_error(exact, estimate):
    if exact != 0:
        return abs(estimate - exact) / exact * 100
    elif estimate == 0:
        return 0
    else:
        return 100


def run_experiment(data_source, dataset_id, sample_size, sampling_method="uniform"):
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
    queries_path = os.path.join(QUERIES_DIR, dataset_full_id, query_set + ".csv")
    models_dir = os.path.join(
        "my_evaluation", "models", dataset_full_id, f"sample_size_{sample_size}"
    )
    results_path = os.path.join(
        "my_evaluation",
        "results",
        dataset_full_id,
        query_set,
        f"sample_size_{sample_size}",
    )
    ground_truth_path = os.path.join("my_evaluation", "ground_truth", dataset_full_id)
    results_filepath = os.path.join(results_path, "results.csv")
    info_filepath = os.path.join(results_path, "info.txt")
    data, n_bytes_per_value = load_dataset(csv_path, dataset_full_id)
    df_queries = pd.read_csv(queries_path, dtype={"predicate_column": int})
    n_rows = schema["n_rows"]
    n_columns = len(schema["column_names"])

    # Ensure not to overwrite output files
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    if os.path.isfile(info_filepath):
        info_filepath = info_filepath.split(".txt")[0] + ts + ".txt"
    if os.path.isfile(results_filepath):
        results_filepath = results_filepath.split(".csv")[0] + ts + ".csv"

    # Create models
    # method <method> <rate> specifies the sampling rate for creating the models.
    # e.g. "method uniform size 1000" does uniform sampling with 1000 samples
    t_modelling_start = perf_counter()
    logger.info("Creating data model...")
    sql_executor = build_models(dataset_full_id, csv_path, models_dir, sample_size)
    t_modelling = perf_counter() - t_modelling_start

    # Compute ground truth
    gt_filename = query_set + "gt.csv"
    results = []
    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)
    if gt_filename not in os.listdir(ground_truth_path):
        logger.info("Computing ground truth")
        for i, query in df_queries.iterrows():
            if (i % 50) == 0:
                logger.info(f"query: {i}")
            predicate_col = int(query.predicate_column)
            predicate_col_name = schema["column_names"][predicate_col]
            predicate = [query.predicate_low, query.predicate_high]
            mask = (data[predicate_col_name] > predicate[0]) & (
                data[predicate_col_name] < predicate[1]
            )
            for aggregation_col in range(n_columns):
                aggregation_col_name = schema["column_names"][aggregation_col]
                for aggregation in AGGREGATIONS:
                    exact = aggregate(data.loc[mask][aggregation_col_name], aggregation)
                    results.append(
                        {
                            "query_id": i,
                            "predicate_column": predicate_col,
                            "aggregation_column": aggregation_col,
                            "aggregation": aggregation,
                            "exact_value": exact,
                        }
                    )
        df_gt = pd.DataFrame(results)
        logger.info("Exporting ground truth.")
        df_gt.to_csv(os.path.join(ground_truth_path, gt_filename), index=False)
    # Load ground truth
    else:
        df_gt = pd.read_csv(os.path.join(ground_truth_path, gt_filename))

    # Run queries
    logger.info("Evaluating queries")
    results = []
    n_queries = 0
    t_queries_start = perf_counter()
    for i, query in df_queries.iterrows():
        predicate_col = int(query.predicate_column)
        predicate_col_name = schema["column_names"][predicate_col]
        predicate = [query.predicate_low, query.predicate_high]
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
                    f"where {predicate[0]} < {predicate_col_name} < {predicate[1]}"
                )
                try:
                    predicted_values, t_estimate = sql_executor.execute(query_str)
                except NotImplementedError:
                    continue
                predicted_value = predicted_values.iloc[0][1]  # extract prediction

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
                    }
                )
                n_queries = n_queries + 1
    t_queries = perf_counter() - t_queries_start

    # Merge with ground truth data
    df = pd.DataFrame(results)
    df = pd.merge(
        df,
        df_gt,
        how="left",
        on=["query_id", "predicate_column", "aggregation_column", "aggregation"],
    )

    # Compute error statistics
    df["error"] = df["predicted_value"] - df["exact_value"]
    df["relative_error"] = df.apply(lambda x: get_relative_error(x.exact_value, x.predicted_value), axis=1)

    # Export results
    logger.info("Exporting results.")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    df.to_csv(results_filepath, index=False)

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
    s_original = n_rows * n_columns * n_bytes_per_value
    s_models = 0
    for file_name in os.listdir(models_dir):
        if file_name.startswith(dataset_full_id) and file_name.endswith(".dill"):
            s_models += os.stat(os.path.join(models_dir, file_name)).st_size

    # Export configuration and performance statistics
    os.makedirs(os.path.dirname(info_filepath), exist_ok=True)
    logger.info(f"Saving experiment parameters and statistics to {info_filepath}")
    n_epoch = sql_executor.get_parameter("n_epoch")
    n_gaussians_reg = sql_executor.get_parameter("n_gaussians_reg")
    n_gaussians_den = sql_executor.get_parameter("n_gaussians_density")
    regression_type = sql_executor.get_parameter("reg_type")
    density_type = sql_executor.get_parameter("density_type")
    device_type = sql_executor.get_parameter("device")
    n_mdn_layer_node_reg = sql_executor.get_parameter("n_mdn_layer_node_reg")
    n_mdn_layer_node_den = sql_executor.get_parameter("n_mdn_layer_node_density")
    integration_epsabs = sql_executor.get_parameter("epsabs")
    integration_epsrel = sql_executor.get_parameter("epsrel")
    integration_n_divisions = sql_executor.get_parameter("n_division")
    integration_limit = sql_executor.get_parameter("limit")
    with open(info_filepath, "w", newline="") as f:
        f.write(f"------------- Parameters -------------\n")
        f.write(f"DATA_SOURCE              {data_source}\n")
        f.write(f"DATASET_ID               {dataset_id}\n")
        f.write(f"QUERIES_SET              {query_set}\n")
        f.write(f"SAMPLING_METHOD          {sampling_method}\n")
        f.write(f"SAMPLE_SIZE              {sample_size:,d}\n")
        f.write(f"N_EPOCH                  {n_epoch:,d}\n")
        f.write(f"N_GAUSSIANS_REG          {n_gaussians_reg:,d}\n")
        f.write(f"N_GAUSSIANS_DEN          {n_gaussians_den:,d}\n")
        f.write(f"N_MDN_LAYER_NODE_REG     {n_mdn_layer_node_reg:,d}\n")
        f.write(f"N_MDN_LAYER_NODE_DEN     {n_mdn_layer_node_den:,d}\n")
        f.write(f"REGRESSION_TYPE          {regression_type}\n")
        f.write(f"DENSITY_TYPE             {density_type}\n")
        f.write(f"DEVICE_TYPE              {device_type}\n")
        f.write(f"INTEGRATION_EPSABS       {integration_epsabs}\n")
        f.write(f"INTEGRATION_EPSREL       {integration_epsrel}\n")
        f.write(f"INTEGRATION_N_DIVISIONS  {integration_n_divisions}\n")
        f.write(f"INTEGRATION_LIMIT        {integration_limit}\n")

        f.write(f"\n------------- Runtime -------------\n")
        f.write(f"Generate models          {t_modelling:.3f} s\n")
        f.write(f"Run queries              {t_queries:.3f} s\n")
        f.write(f"Queries executed         {n_queries}\n")
        if n_queries:
            mean_latency = t_queries / n_queries
        else:
            mean_latency = 0
        f.write(f"Mean latency             {mean_latency:.6f} s\n")

        f.write(f"\n------------- Storage -------------\n")
        f.write(f"Original data            {s_original:,d} bytes\n")
        f.write(f"Models                   {s_models:,d} bytes\n")
        f.write(f"Models (%)               {s_models / s_original * 100:.2f} %\n")
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
    run_experiment("kaggle", "light_detection", 1000)
    # run_experiment("uci", "household_power_consumption", 1000)
    # run_experiment("uci", "gas_sensor_home_activity", 1000)

    # run_experiment("kaggle", "light_detection", 1000)


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    logging.getLogger("gensim.models.word2vec").setLevel(logging.WARNING)
    logging.getLogger("gensim.utils").setLevel(logging.WARNING)
    main()