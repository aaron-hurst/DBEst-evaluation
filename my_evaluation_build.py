"""Build and save DBEst++ models."""

import json
import logging

# import multiprocessing
import os
from datetime import datetime

# from itertools import repeat
from time import perf_counter

import numpy as np
import pandas as pd

from dbestclient.executor.executor import SqlExecutor
from config import LOG_FORMAT, RESULTS_DIR, DATA_DIR

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

DUMMY_COLUMN_NAME = "_group"
DUMMY_COLUMN_TEXT = "all"

CHUNK_SIZE = 10000000
SAMPLE_SIZE = 10000
SAVE_SAMPLE = True
ONLY_REQUIRED_MODELS = True
SAMPLING_METHOD = "uniform"
SUFFIXES = ["_10m", "_100m", "_1b"]

logger = logging.getLogger(__name__)


def load_sample(filepath, total_rows, sample_size=10000, header=0, chunk_size=10000000):
    """Loads a dataset from a CSV file. Loads in batches if the file is vary large."""

    # Number of samples to extract for each chunk
    n_chunks = int(np.ceil(total_rows / chunk_size))
    sample_ratio = sample_size / total_rows
    if total_rows % chunk_size:
        chunk_sizes = [chunk_size] * (n_chunks - 1) + [total_rows % chunk_size]
    else:
        chunk_sizes = [chunk_size] * (n_chunks - 1) + [chunk_size]
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
    dataset_id,
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
    table_name = (f"{dataset_id}_{col_1}_" f"{col_2}_{sample_size}").replace("-", "_")
    sql_create_model = (
        f"create table "
        f"{table_name}({column_1_str}, {column_2_str}) "
        f"from '{sample_filepath}' "
        f"group by {DUMMY_COLUMN_NAME} "
        f"method {SAMPLING_METHOD} size {sample_size};"
    )
    t_build_model_start = perf_counter()
    logger.info(f"Builing model: {table_name}...")
    try:
        sql_executor.execute(sql_create_model)
    except FileExistsError:
        logger.info(f"Model already exists: {table_name}")
        return None
    except NotImplementedError as e:
        logger.info(f"Failed to build model {table_name}: {e}")
        return None
    return perf_counter() - t_build_model_start


def build_model_for_dataset(dataset_id, query_set):
    logger.info(f"Analysing dataset: {dataset_id}")
    logger.info(f"Sample size: {SAMPLE_SIZE}")

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_id_no_suffix = dataset_id
    for suffix in SUFFIXES:
        dataset_id_no_suffix = dataset_id_no_suffix.removesuffix(suffix)
    output_dir = os.path.join(RESULTS_DIR, "aqp", "dbestpp")
    schema_filepath = os.path.join(
        DATA_DIR, "schemas", "aqp", f"{dataset_id_no_suffix}.json"
    )
    data_filepath = os.path.join(DATA_DIR, "processed", f"{dataset_id}.csv")
    sample_filepath = os.path.join(
        output_dir, "data", f"{dataset_id}__sample_size_{SAMPLE_SIZE}.csv"
    )
    models_dir = os.path.join(
        output_dir, "models", f"{dataset_id}_sample_size_{SAMPLE_SIZE}_{timestamp}"
    )
    metadata_filepath = os.path.join(models_dir, "build_metadata.txt")
    n = sum(1 for _ in open(data_filepath)) - 1  # excludes header
    logger.info(f"Total rows: {n}")

    # Load list of required models
    required_models = None
    if ONLY_REQUIRED_MODELS:
        logger.info("Loading required models list...")
        required_models_filepath = os.path.join(
            "evaluation", f"models_required_{dataset_id}_v{query_set}.txt"
        )
        with open(required_models_filepath, "r") as fp:
            required_models = fp.readlines()
        required_models = [
            "_".join(x.split(",")).replace("\n", "") for x in required_models
        ]

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
        df = load_sample(
            data_filepath, n, sample_size=SAMPLE_SIZE, chunk_size=CHUNK_SIZE
        )
        df[DUMMY_COLUMN_NAME] = DUMMY_COLUMN_TEXT
        os.makedirs(os.path.dirname(sample_filepath), exist_ok=True)
        df.to_csv(sample_filepath, index=False)

    # Generate models
    logger.info("Creating data models...")
    os.makedirs(models_dir, exist_ok=True)
    with open(schema_filepath, "r") as f:
        schema = json.load(f)
    t_modelling_start = perf_counter()
    n_cols = len(schema["column_names"])
    for i in range(n_cols):
        for j in range(n_cols):
            col_name_1 = schema["column_names"][i]
            col_name_2 = schema["column_names"][j]
            if (required_models is None) or (
                required_models and (f"{col_name_1}_{col_name_2}" in required_models)
            ):
                build_model(
                    dataset_id,
                    col_name_1,
                    col_name_2,
                    schema["sql_types"][i],
                    schema["sql_types"][j],
                    sample_filepath,
                    models_dir,
                    n,
                )
    # t_modelling_sum = 0
    # with multiprocessing.Pool() as pool:
    #     model_timings = pool.starmap(
    #         build_model,
    #         zip(
    #             np.repeat(schema["column_names"], n_cols),
    #             np.tile(schema["column_names"], n_cols),
    #             np.repeat(schema["sql_types"], n_cols),
    #             np.tile(schema["sql_types"], n_cols),
    #             repeat(sample_filepath),
    #             repeat(models_dir),
    #             repeat(n),
    #         ),
    #     )
    # t_modelling_sum += sum([t for t in model_timings if t is not None])
    t_modelling_real = perf_counter() - t_modelling_start

    # Get total size of models
    n_models = 0
    s_models = 0
    for f in os.listdir(models_dir):
        if f.startswith(dataset_id.replace("-", "_")) and f.endswith(".dill"):
            s_models += os.stat(os.path.join(models_dir, f)).st_size
            n_models += 1

    # Export parameters and statistics
    logger.info("Exporting metadata...")
    sql_executor = SqlExecutor(models_dir, save_sample=SAVE_SAMPLE)
    density_type = sql_executor.get_parameter("density_type")
    device_type = sql_executor.get_parameter("device")
    integration_epsabs = sql_executor.get_parameter("epsabs")
    integration_epsrel = sql_executor.get_parameter("epsrel")
    integration_limit = sql_executor.get_parameter("limit")
    integration_n_divisions = sql_executor.get_parameter("n_division")
    n_embedding_dim = sql_executor.get_parameter("n_embedding_dim")
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
        f.write(f"dataset_id                {dataset_id}\n")
        f.write(f"SAMPLE_SIZE               {SAMPLE_SIZE}\n")
        f.write(f"SAMPLING_METHOD           {SAMPLING_METHOD}\n")
        f.write(f"DENSITY_TYPE              {density_type}\n")
        f.write(f"DEVICE_TYPE               {device_type}\n")
        f.write(f"INTEGRATION_EPSABS        {integration_epsabs}\n")
        f.write(f"INTEGRATION_EPSREL        {integration_epsrel}\n")
        f.write(f"INTEGRATION_LIMIT         {integration_limit}\n")
        f.write(f"INTEGRATION_N_DIVISIONS   {integration_n_divisions}\n")
        f.write(f"N_EMBEDDING_DIM           {n_embedding_dim}\n")
        f.write(f"N_EPOCH                   {n_epoch}\n")
        f.write(f"N_GAUSSIANS_DEN           {n_gaussians_den}\n")
        f.write(f"N_GAUSSIANS_REG           {n_gaussians_reg}\n")
        f.write(f"N_MDN_LAYER_NODE_DEN      {n_mdn_layer_node_den}\n")
        f.write(f"N_MDN_LAYER_NODE_REG      {n_mdn_layer_node_reg}\n")
        f.write(f"REGRESSION_TYPE           {regression_type}\n")
        f.write(f"WORD2VEC_MIN_COUNT        {word2vec_min_count}\n")
        f.write(f"WORD2VEC_EPOCHS           {word2vec_epochs}\n")

        f.write("\n------------- Runtime -------------\n")
        # f.write(f"Generate models (sum)     {t_modelling_sum:.3f} s\n")
        f.write(f"Generate models (real)    {t_modelling_real:.3f} s\n")

        f.write("\n------------- Storage -------------\n")
        f.write(f"Total models              {n_models:,d} bytes\n")
        f.write(f"Models storage            {s_models:,d} bytes\n")

    logger.info("Done.")


def main():
    for dataset_id, query_set in DATASETS.items():
        build_model_for_dataset(dataset_id, query_set)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.getLogger("gensim.models.word2vec").setLevel(logging.ERROR)
    logging.getLogger("gensim.models.fasttext").setLevel(logging.ERROR)
    logging.getLogger("gensim.utils").setLevel(logging.ERROR)
    main()
