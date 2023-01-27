from time import perf_counter
import json
import logging
import os

from dbestclient.executor.executor import SqlExecutor

from config import LOG_FORMAT, NAME_DELIMITER, DATA_DIR, WAREHOUSE_DIR
from schemas import DB_SCHEMAS

LOGGING_LEVEL = logging.INFO
DATA_MODEL_CREATED = False
DATA_SOURCE = "uci"
DATASET_ID = "household_power_consumption"
QUERIES_SET = "uci-household_power_consumption-N=100"
SAMPLING_METHOD = "uniform"
SAMPLE_SIZE = 1000
SAVE_SAMPLE = True
# 'verbose': True,
# 'b_show_latency': True,
# 'backend_server': None,
# 'epsabs': 10.0,
# 'epsrel': 0.1,
# 'mesh_grid_num': 20,
# 'limit': 30,
# 'csv_split_char': ',',
# "num_epoch":400,
# "reg_type":"mdn",


def build_models(dataset_full_id, csv_path):
    """Build models for each pair of columns."""
    table_name = dataset_full_id  # .replace(NAME_DELIMITER, "_")
    schema = DB_SCHEMAS[DATA_SOURCE][DATASET_ID]
    sqlExecutor = SqlExecutor(WAREHOUSE_DIR, SAVE_SAMPLE)
    n_columns = len(schema["column_names"])
    for i in range(n_columns):
        column_1_str = schema["column_names"][i] + " " + schema["column_types"][i]
        for j in range(n_columns):
            logger.debug(f"Building model for column pair: ({i}, {j})")
            column_2_str = schema["column_names"][j] + " " + schema["column_types"][j]
            sql_create_model = (
                f"create table "
                f"{table_name}({column_1_str}, {column_2_str}) "
                f"from '{csv_path}' "
                f"method {SAMPLING_METHOD} size {SAMPLE_SIZE};"
            )
            sqlExecutor.execute(sql_create_model)
    return


def main():
    # Setup
    dataset_full_id = DATA_SOURCE + NAME_DELIMITER + DATASET_ID
    csv_path = os.path.join(DATA_DIR, "uncompressed", dataset_full_id + ".csv")
    results_path = os.path.join(
        "experiments",
        "comparisons_for_paper",
        dataset_full_id,
        QUERIES_SET + "_" + str(SAMPLE_SIZE),
    )
    results_filepath = os.path.join(results_path, "results.csv")
    info_filepath = os.path.join(results_path, "info.txt")
    sqlExecutor = SqlExecutor(WAREHOUSE_DIR)

    # Ensure warehouse directory exists
    if os.path.exists(WAREHOUSE_DIR):
        logger.info("Warehouse is initialized.")
    else:
        logger.info("Warehouse does not exists, so initialize one.")
        os.mkdir(WAREHOUSE_DIR)

    # Create models
    # method <method> <rate> specifies the sampling rate for creating the models.
    # e.g. "method uniform size 1000" does uniform sampling with 1000 samples
    t_modelling_start = perf_counter()
    if not DATA_MODEL_CREATED:
        logger.info("Creating data model...")
        build_models(dataset_full_id, csv_path)
    t_modelling = perf_counter() - t_modelling_start

    # Run queries
    # TODO
    # include query id, aggregation, latency, predicted result, error, relative error,
    # confidence interval (?), relative confidence interval (half width)
    logger.info("Evaluating queries")
    t_queries_start = perf_counter()
    n_queries = 0
    t_queries = perf_counter() - t_queries_start

    # Export results
    # TODO

    # Get storage information
    s_original = 1
    s_models = 0  # os.stat(os.path.join(hdf_path, DATASET_ID + ".hdf")).st_size

    # Export configuration and performance statistics
    # TODO finishing touches
    os.makedirs(os.path.dirname(info_filepath), exist_ok=True)
    logger.info(f"Saving experiment parameters and statistics to {info_filepath}")
    with open(info_filepath, "w", newline="") as f:
        f.write(f"------------- Parameters -------------\n")
        f.write(f"DATA_MODEL_CREATED           {DATA_MODEL_CREATED}\n")
        f.write(f"DATA_SOURCE                  {DATA_SOURCE}\n")
        f.write(f"DATASET_ID                   {DATASET_ID}\n")
        f.write(f"QUERIES_SET                  {QUERIES_SET}\n")
        f.write(f"SAMPLING_METHOD              {SAMPLING_METHOD}\n")
        f.write(f"SAMPLE_SIZE                  {SAMPLE_SIZE}\n")
        # f.write(f"CONFIDENCE_INTERVAL_ALPHA    {CONFIDENCE_INTERVAL_ALPHA}\n")
        # f.write(f"BLOOM_FILTERS                {BLOOM_FILTERS}\n")
        # f.write(f"RDC_THRESHOLD                {RDC_THRESHOLD}\n")
        # f.write(f"POST_SAMPLING_FACTOR         {POST_SAMPLING_FACTOR}\n")
        # f.write(f"INCREMENTAL_LEARNING_RATE    {INCREMENTAL_LEARNING_RATE}\n")
        # f.write(f"RDC_SPN_SELECTION            {RDC_SPN_SELECTION}\n")
        # f.write(f"PAIRWISE_RDC_PATH            {PAIRWISE_RDC_PATH}\n")
        # f.write(f"MAX_VARIANTS                 {MAX_VARIANTS}\n")
        # f.write(f"MERGE_INDICATOR_EXP          {MERGE_INDICATOR_EXP}\n")
        # f.write(f"EXPLOIT_OVERLAPPING          {EXPLOIT_OVERLAPPING}\n")
        # f.write(f"SHOW_CONFIDENCE_INTERVALS    {SHOW_CONFIDENCE_INTERVALS}\n")

        f.write(f"\n------------- Runtime -------------\n")
        f.write(f"Generate models              {t_modelling:.3f} s\n")
        f.write(f"Run queries                  {t_queries:.3f} s\n")
        f.write(f"Queries executed             {n_queries}\n")
        f.write(f"Mean latency                 {t_queries / n_queries:.6f} s\n")

        f.write(f"\n------------- Storage -------------\n")
        f.write(f"Original data                {s_original:,d}\n")
        f.write(f"HDF files                    {s_models:,d}\n")
        f.write(f"SPN ensembles                {s_models:,d}\n")
        f.write(f"SPN ensembles (%)            {s_models / s_original * 100:.2f} %\n")

    # ex = SqlExecutor()
    # # )
    # ex.execute("set n_mdn_layer_node_reg=10")
    # ex.execute("set n_mdn_layer_node_density=15")
    # ex.execute("set n_jobs=1")
    # ex.execute("set n_hidden_layer=1")
    # ex.execute("set n_epoch=20")
    # ex.execute("set n_gaussians_reg=8")
    # ex.execute("set n_gaussians_density=10")
    # ex.execute("set csv_split_char=','")
    # ex.execute("set table_header='No,year,month,day,hour,pm25,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir'")
    # ex.execute(
    #     "create table pm25template(pm25 real, PRES real) from dbestwarehouse/pm25.csv GROUP BY PRES"  # method uniform size 0.001"
    # )
    # # ex.execute(
    # #     "create table pm25template(pm25 real, year real) from 'dbestwarehouse/pm25.csv' GROUP BY year method uniform size 0.001 "
    # # )


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    main()
