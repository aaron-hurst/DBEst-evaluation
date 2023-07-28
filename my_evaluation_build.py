"""Build and save DBEst++ models."""

import json
import logging
import os
from time import perf_counter

import numpy as np
import pandas as pd

from dbestclient.executor.executor import SqlExecutor
from config import LOG_FORMAT, RESULTS_DIR, DATA_DIR


DATASET_ID = "uci-household_power_consumption"

DUMMY_COLUMN_NAME = "_group"
DUMMY_COLUMN_TEXT = "all"

SAMPLE_SIZE = 10000
SAVE_SAMPLE = True
SAMPLING_METHOD = "uniform"


def main():
    logger.info(f"Analysing dataset: {DATASET_ID}")
    logger.info(f"Sample size: {SAMPLE_SIZE}")

    # Setup
    output_dir = os.path.join(RESULTS_DIR, "aqp", "dbestpp")
    schema_filepath = os.path.join(DATA_DIR, "schemas", "aqp", f"{DATASET_ID}.json")

    sample_filepath = os.path.join(
        output_dir, "data", f"{DATASET_ID}__sample_size_{SAMPLE_SIZE}.csv"
    )
    models_dir = os.path.join(
        output_dir, "models", f"{DATASET_ID}_sample_size_{SAMPLE_SIZE}"
    )
    metadata_filepath = os.path.join(models_dir, "build_metadata.txt")

    # If not already created, extract a random sample from the dataset and save it as a
    # CSV file with an additional "dummy" group column. DBEst++ uses the group column
    # for query execution. DBEst++ does not work correctly on queries (and models) that
    # do not have a group by clause. I don't understand why not. DBEst++ does insert a
    # "dummy" column for models/queries without a group by clause, however the query
    # results are always very poor for COUNT, SUM and VARIANCE queries and often poor
    # for AVG queries. Manually adding a "group" column that is only contains a single
    # value will hopefully work as a hack solution to get DBEst++ to perform reasonably
    # well on queries without a group by clause.
    # NOTE: A separate file is created for each sample size used.
    if not os.path.isfile(sample_filepath):
        logger.info("Generating sample file with dummy group by column...")
        data_filepath = os.path.join(DATA_DIR, "processed", f"{DATASET_ID}.csv")
        n = sum(1 for _ in open(data_filepath)) - 1  # excludes header
        skip_rows = np.sort(
            np.random.choice(np.arange(1, n + 1), n - SAMPLE_SIZE, replace=False)
        )
        df = pd.read_csv(data_filepath, header=0, skiprows=skip_rows)
        df[DUMMY_COLUMN_NAME] = DUMMY_COLUMN_TEXT
        os.makedirs(os.path.dirname(sample_filepath), exist_ok=True)
        df.to_csv(sample_filepath, index=False)

    # Generate models
    logger.info("Creating data models...")
    os.makedirs(models_dir, exist_ok=True)
    with open(schema_filepath, "r") as f:
        schema = json.load(f)
    t_modelling_start = perf_counter()
    sql_executor = SqlExecutor(models_dir, save_sample=SAVE_SAMPLE)
    n_columns = len(schema["column_names"])
    for i in range(n_columns):
        for j in range(n_columns):
            column_1_str = f"{schema['column_names'][i]} {schema['sql_types'][i]}"
            column_2_str = f"{schema['column_names'][j]} {schema['sql_types'][j]}"
            table_name = (
                f"{DATASET_ID}_{schema['column_names'][i]}_"
                f"{schema['column_names'][j]}_{SAMPLE_SIZE}"
            ).replace("-", "_")
            sql_create_model = (
                f"create table "
                f"{table_name}({column_1_str}, {column_2_str}) "
                f"from '{sample_filepath}' "
                f"group by {DUMMY_COLUMN_NAME} "
                f"method {SAMPLING_METHOD} size {SAMPLE_SIZE};"
            )
            try:
                sql_executor.execute(sql_create_model)
                logger.debug(f"Model built for column pair ({i}, {j}).")
            except FileExistsError:
                logger.debug(f"Model already exists for column pair ({i}, {j}).")
                continue
            except (ValueError, IndexError, TypeError) as e:
                logger.error(f"Failed to build model for column pair ({i}, {j}): {e}")
                continue
    t_modelling = perf_counter() - t_modelling_start

    # Get total size of models
    s_models = 0
    for f in os.listdir(models_dir):
        if f.startswith(DATASET_ID) and f.endswith(".dill"):
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
    with open(metadata_filepath, "w", newline="") as f:
        f.write("------------- Parameters -------------\n")
        f.write(f"DATASET_ID                {DATASET_ID}\n")
        f.write(f"SAMPLING_METHOD           {SAMPLING_METHOD}\n")
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
        f.write(f"Generate models           {t_modelling:.3f} s\n")

        f.write("\n------------- Storage -------------\n")
        f.write(f"Models                    {s_models:,d} bytes\n")

    logger.info("Done.")
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    logging.getLogger("gensim.models.base_any2vec").setLevel(logging.ERROR)
    logging.getLogger("gensim.models.word2vec").setLevel(logging.ERROR)
    logging.getLogger("gensim.utils").setLevel(logging.ERROR)
    main()
