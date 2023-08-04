"""Build and save DBEst++ models."""

import json
import logging
import multiprocessing
import os
from itertools import repeat
from time import perf_counter

import numpy as np
import pandas as pd

from dbestclient.executor.executor import SqlExecutor
from config import LOG_FORMAT, RESULTS_DIR, DATA_DIR


DATASET_ID = "uci-household_power_consumption"
# DATASET_ID = "usdot-flights_10m"

DUMMY_COLUMN_NAME = "_group"
DUMMY_COLUMN_TEXT = "all"

CHUNK_SIZE = 10000000
SAMPLE_SIZE = 1000
SAVE_SAMPLE = True
SAMPLING_METHOD = "uniform"
SUFFIXES = ["_10m", "_100m", "_1b"]

logger = logging.getLogger(__name__)


def load_sample(filepath, total_rows, sample_size=10000, header=0, chunk_size=10000000):
    """Loads a dataset from a CSV file. Loads in batches if the file is vary large."""

    # Number of samples to extract for each chunk
    n_chunks = int(np.ceil(total_rows / chunk_size))
    sample_ratio = sample_size / total_rows
    chunk_sizes = [chunk_size] * (n_chunks - 1) + [total_rows % chunk_size]
    samples_per_chunk = [int(sample_ratio * s) for s in chunk_sizes]
    if sum(samples_per_chunk) < sample_size:
        diff = sample_size - sum(samples_per_chunk)
        samples_per_chunk = [s + (i < diff) for i, s in enumerate(samples_per_chunk)]

    # Run sampling
    logger.info(f"Total chunks: {n_chunks} (chunk size={chunk_size})")
    df_chunks = []
    with pd.read_csv(filepath, header=header, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            logger.info(f"Loading chunk {i+1}/{n_chunks}")
            sample_rows = np.sort(
                np.random.choice(
                    np.arange(chunk.shape[0]), samples_per_chunk[i], replace=False
                )
            )
            df_chunks.append(chunk.iloc[sample_rows])

    return pd.concat(df_chunks, axis=0, ignore_index=True, copy=False)


def build_model(
    col_1,
    col_2,
    type_1,
    type_2,
    sample_filepath,
    models_dir,
    n,
    sample_size=SAMPLE_SIZE,
):
    """Build and save a single model for a given column pair."""
    sql_executor = SqlExecutor(models_dir, save_sample=SAVE_SAMPLE)
    sql_executor.n_total_records = {"total": n}
    column_1_str = f"{col_1} {type_1}"
    column_2_str = f"{col_2} {type_2}"
    table_name = (f"{DATASET_ID}_{col_1}_" f"{col_2}_{sample_size}").replace("-", "_")
    sql_create_model = (
        f"create table "
        f"{table_name}({column_1_str}, {column_2_str}) "
        f"from '{sample_filepath}' "
        f"group by {DUMMY_COLUMN_NAME} "
        f"method {SAMPLING_METHOD} size {sample_size};"
    )
    t_build_model_start = perf_counter()
    try:
        sql_executor.execute(sql_create_model)
    except FileExistsError:
        logger.info(f"Model already exists: {table_name}")
        return None
    except NotImplementedError as e:
        logger.info(f"Failed to build model {table_name}: {e}")
        return None
    logger.info(f"Built model: {table_name}")
    return perf_counter() - t_build_model_start


def main():
    logger.info(f"Analysing dataset: {DATASET_ID}")
    logger.info(f"Sample size: {SAMPLE_SIZE}")

    # Setup
    dataset_id_no_suffix = DATASET_ID
    for suffix in SUFFIXES:
        dataset_id_no_suffix = dataset_id_no_suffix.removesuffix(suffix)
    output_dir = os.path.join(RESULTS_DIR, "aqp", "dbestpp")
    schema_filepath = os.path.join(
        DATA_DIR, "schemas", "aqp", f"{dataset_id_no_suffix}.json"
    )
    data_filepath = os.path.join(DATA_DIR, "processed", f"{DATASET_ID}.csv")
    sample_filepath = os.path.join(
        output_dir, "data", f"{DATASET_ID}__sample_size_{SAMPLE_SIZE}.csv"
    )
    models_dir = os.path.join(
        output_dir, "models", f"{DATASET_ID}_sample_size_{SAMPLE_SIZE}"
    )
    metadata_filepath = os.path.join(models_dir, "build_metadata.txt")
    n = sum(1 for _ in open(data_filepath)) - 1  # excludes header
    logger.info(f"Total rows: {n}")

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
        df = load_sample(data_filepath, n, SAMPLE_SIZE, chunk_size=CHUNK_SIZE)
        df[DUMMY_COLUMN_NAME] = DUMMY_COLUMN_TEXT
        os.makedirs(os.path.dirname(sample_filepath), exist_ok=True)
        df.to_csv(sample_filepath, index=False)

    # Generate models
    logger.info("Creating data models...")
    os.makedirs(models_dir, exist_ok=True)
    with open(schema_filepath, "r") as f:
        schema = json.load(f)
    t_modelling_start = perf_counter()
    t_modelling_sum = 0
    n_cols = len(schema["column_names"])
    with multiprocessing.Pool() as pool:
        model_timings = pool.starmap(
            build_model,
            zip(
                np.repeat(schema["column_names"], n_cols),
                np.tile(schema["column_names"], n_cols),
                np.repeat(schema["sql_types"], n_cols),
                np.tile(schema["sql_types"], n_cols),
                repeat(sample_filepath),
                repeat(models_dir),
                repeat(n),
            ),
        )
    t_modelling_sum += sum([t for t in model_timings if t is not None])
    t_modelling_real = perf_counter() - t_modelling_start

    # Get total size of models
    s_models = 0
    for f in os.listdir(models_dir):
        if f.startswith(DATASET_ID.replace("-", "_")) and f.endswith(".dill"):
            s_models += os.stat(os.path.join(models_dir, f)).st_size

    # Export parameters and statistics
    logger.info("Exporting metadata...")
    sql_executor = SqlExecutor(models_dir, save_sample=SAVE_SAMPLE)
    density_type = sql_executor.get_parameter("density_type")
    device_type = sql_executor.get_parameter("device")
    integration_epsabs = sql_executor.get_parameter("epsabs")
    integration_epsrel = sql_executor.get_parameter("epsrel")
    integration_limit = sql_executor.get_parameter("limit")
    integration_n_divisions = sql_executor.get_parameter("n_division")
    n_epoch = sql_executor.get_parameter("n_epoch")
    n_gaussians_den = sql_executor.get_parameter("n_gaussians_density")
    n_gaussians_reg = sql_executor.get_parameter("n_gaussians_reg")
    n_mdn_layer_node_den = sql_executor.get_parameter("n_mdn_layer_node_density")
    n_mdn_layer_node_reg = sql_executor.get_parameter("n_mdn_layer_node_reg")
    regression_type = sql_executor.get_parameter("reg_type")
    word2vec_min_count = sql_executor.get_parameter("word2vec_min_count")
    word2vec_epochs = sql_executor.get_parameter("word2vec_epochs")
    with open(metadata_filepath, "w", newline="") as f:
        f.write("------------- Parameters -------------\n")
        f.write(f"DATASET_ID                {DATASET_ID}\n")
        f.write(f"SAMPLE_SIZE               {SAMPLE_SIZE}\n")
        f.write(f"SAMPLING_METHOD           {SAMPLING_METHOD}\n")
        f.write(f"DENSITY_TYPE              {density_type}\n")
        f.write(f"DEVICE_TYPE               {device_type}\n")
        f.write(f"INTEGRATION_EPSABS        {integration_epsabs}\n")
        f.write(f"INTEGRATION_EPSREL        {integration_epsrel}\n")
        f.write(f"INTEGRATION_LIMIT         {integration_limit}\n")
        f.write(f"INTEGRATION_N_DIVISIONS   {integration_n_divisions}\n")
        f.write(f"N_EPOCH                   {n_epoch}\n")
        f.write(f"N_GAUSSIANS_DEN           {n_gaussians_den}\n")
        f.write(f"N_GAUSSIANS_REG           {n_gaussians_reg}\n")
        f.write(f"N_MDN_LAYER_NODE_DEN      {n_mdn_layer_node_den}\n")
        f.write(f"N_MDN_LAYER_NODE_REG      {n_mdn_layer_node_reg}\n")
        f.write(f"REGRESSION_TYPE           {regression_type}\n")
        f.write(f"WORD2VEC_MIN_COUNT        {word2vec_min_count}\n")
        f.write(f"WORD2VEC_EPOCHS           {word2vec_epochs}\n")

        f.write("\n------------- Runtime -------------\n")
        f.write(f"Generate models (sum)     {t_modelling_sum:.3f} s\n")
        f.write(f"Generate models (real)    {t_modelling_real:.3f} s\n")

        f.write("\n------------- Storage -------------\n")
        f.write(f"Models                    {s_models:,d} bytes\n")

    logger.info("Done.")
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.getLogger("gensim.models.word2vec").setLevel(logging.ERROR)
    logging.getLogger("gensim.utils").setLevel(logging.ERROR)
    main()
