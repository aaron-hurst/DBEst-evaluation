"""Run queries using existing DBEst++ models."""

import logging
import os
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd

from dbestclient.executor.executor import SqlExecutor
from config import LOG_FORMAT, RESULTS_DIR, QUERIES_DIR


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
SAMPLE_SIZE = 10000
INCLUDE_FAILED = False


def get_relative_error_pct(true, predicted):
    if np.isnan(predicted):
        return 100
    elif true != 0:
        return abs((predicted - true) / true) * 100
    elif predicted == 0:
        return 0
    else:
        return 100


def get_relative_bound_pct(true, ci_half_width):
    if np.isnan(ci_half_width):
        return 100
    elif true != 0:
        return abs(ci_half_width / true) * 100
    elif ci_half_width == 0:
        return 0
    else:
        return 100


def run_evaluation(dataset_id, query_set):
    logger.info(f"Analysing dataset: {dataset_id}")
    logger.info(f"Sample size: {SAMPLE_SIZE}")

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dbestpp_dir = os.path.join(RESULTS_DIR, "aqp", "dbestpp")
    queries_filepath = os.path.join(QUERIES_DIR, f"{dataset_id}_v{query_set}.txt")
    ground_truth_filepath = os.path.join(
        QUERIES_DIR, "ground_truth", f"{dataset_id}_v{query_set}_gt.csv"
    )
    results_filepath = os.path.join(
        dbestpp_dir,
        "results",
        dataset_id,
        f"queries_v{query_set}_sample_size_{SAMPLE_SIZE}_{timestamp}.csv",
    )
    metadata_filepath = os.path.join(
        dbestpp_dir,
        "results",
        dataset_id,
        f"queries_v{query_set}_sample_size_{SAMPLE_SIZE}_{timestamp}_metadata.txt",
    )

    # Get most recent models directory fulfilling criteria
    models_dir = None
    model_prefix = f"{dataset_id}_sample_size_{SAMPLE_SIZE}"
    for dirname in os.listdir(os.path.join(dbestpp_dir, "models")):
        if dirname == model_prefix:
            models_dir = os.path.join(dbestpp_dir, "models", dirname)
            break
    if models_dir is None:
        raise FileNotFoundError(f"No model found for {model_prefix}")
    logger.info(f"Using models from: {models_dir}")

    # Evaluate queries
    logger.info("Evaluating queries...")
    with open(queries_filepath) as f:
        queries = f.readlines()
    results = []
    n_queries_total = len(queries)
    n_queries_executed = 0
    sql_executor = SqlExecutor(models_dir)
    t_queries_start = perf_counter()
    for i, query in enumerate(queries):
        if (i % 100 == 0) and (i > 0):
            logger.info(f"Processed {i}/{n_queries_total} queries")

        # Transform query into a format suitable for DBEst++
        # aggregation column
        # get all predicate columns
        # if more than 2 columns overall, fail
        # else, define model and replace this part of the query
        query_split = query.split(" ")
        aggregation_column = query_split[1].split("(")[1].split(")")[0]
        predicate_columns = [
            query_split[i + 1]
            for i in range(len(query_split))
            if query_split[i].lower() in ["where", "and", "or"]
        ]
        total_columns = len(set([aggregation_column, *predicate_columns]))
        if total_columns > 2:
            logger.info(f"Failed query {i}: more than two columns: {total_columns}.")
            continue
        table = query_split[3]
        model = f"{table}_{aggregation_column}_{predicate_columns[0]}_{SAMPLE_SIZE}"
        query_split[3] = model
        query_dbestpp = " ".join(query_split)
        try:
            estimate, t_estimate = sql_executor.execute(query_dbestpp)
            estimate = estimate.iloc[0, 1]
        except NotImplementedError as e:
            logger.info(f"Failed query {i}: {e}")
            if INCLUDE_FAILED:
                estimate = None
                t_estimate = None
            else:
                continue
        results.append(
            {
                "query_id": i,
                "latency": t_estimate,
                "estimate": estimate,
                "bound_low": None,
                "bound_high": None,
            }
        )
        n_queries_executed = n_queries_executed + 1
    t_queries = perf_counter() - t_queries_start

    # Merge with ground truth data
    logger.info("Merge with ground truth and compute error...")
    df = pd.DataFrame(results)
    df_gt = pd.read_csv(ground_truth_filepath)
    df = pd.merge(df, df_gt, how="left", on=["query_id"])

    # Compute error and bounds statistics
    df["error"] = df["estimate"] - df["exact_result"]
    df["error_relative_pct"] = df.apply(
        lambda x: get_relative_error_pct(x.exact_result, x.estimate), axis=1
    )
    df["bounds_width"] = df["bound_high"] - df["bound_low"]
    df["bounds_width_relative_pct"] = df.apply(
        lambda x: get_relative_bound_pct(x.exact_result, x.bounds_width),
        axis=1,
    )
    df["bound_is_accurate"] = (df["bound_high"] >= df["exact_result"]) & (
        df["bound_low"] <= df["exact_result"]
    )

    # Export results
    logger.info("Exporting results...")
    os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
    df.to_csv(results_filepath, index=False)

    # Get total size of models
    s_models = 0
    for f in os.listdir(models_dir):
        if f.startswith(dataset_id.replace("-", "_")) and f.endswith(".dill"):
            s_models += os.stat(os.path.join(models_dir, f)).st_size

    # Export parameters and statistics
    logger.info("Exporting metadata...")
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
    if n_queries_executed:
        mean_latency = t_queries / n_queries_executed
    else:
        mean_latency = 0
    with open(metadata_filepath, "w", newline="") as f:
        f.write("------------- Parameters -------------\n")
        f.write(f"DATASET_ID                {dataset_id}\n")
        f.write(f"QUERIES_SET               {query_set}\n")
        f.write(f"SAMPLE_SIZE               {SAMPLE_SIZE}\n")
        f.write(f"N_EPOCH                   {n_epoch}\n")
        f.write(f"N_GAUSSIANS_REG           {n_gaussians_reg}\n")
        f.write(f"N_GAUSSIANS_DEN           {n_gaussians_den}\n")
        f.write(f"N_MDN_LAYER_NODE_REG      {n_mdn_layer_node_reg}\n")
        f.write(f"N_MDN_LAYER_NODE_DEN      {n_mdn_layer_node_den}\n")
        f.write(f"REGRESSION_TYPE           {regression_type}\n")
        f.write(f"DENSITY_TYPE              {density_type}\n")
        f.write(f"DEVICE_TYPE               {device_type}\n")
        f.write(f"INTEGRATION_EPSABS        {integration_epsabs}\n")
        f.write(f"INTEGRATION_EPSREL        {integration_epsrel}\n")
        f.write(f"INTEGRATION_N_DIVISIONS   {integration_n_divisions}\n")
        f.write(f"INTEGRATION_LIMIT         {integration_limit}\n")

        f.write("\n------------- Runtime -------------\n")
        f.write(f"Run queries               {t_queries:.3f} s\n")
        f.write(f"Queries total             {n_queries_total}\n")
        f.write(f"Queries executed          {n_queries_executed}\n")
        f.write(f"Mean latency              {mean_latency:.6f} s\n")

        f.write("\n------------- Storage -------------\n")
        f.write(f"Models                    {s_models} bytes\n")

    logger.info("Done.")
    return


def main():
    for dataset_id, query_set in DATASETS.items():
        run_evaluation(dataset_id, query_set)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    # logging.getLogger("gensim.models.base_any2vec").setLevel(logging.ERROR)
    # logging.getLogger("gensim.models.word2vec").setLevel(logging.ERROR)
    # logging.getLogger("gensim.utils").setLevel(logging.ERROR)
    main()
