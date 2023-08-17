import os

from config import QUERIES_DIR

DATASET_ID = "usdot-flights_1b"
QUERY_SET = 4


queries_filepath = os.path.join(QUERIES_DIR, f"{DATASET_ID}_v{QUERY_SET}.txt")
with open(queries_filepath) as f:
    queries = f.readlines()
out_filename = os.path.join(f"models_required_{DATASET_ID}_v{QUERY_SET}.txt")

for i, query in enumerate(queries):
    # Multi-column queries
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

    # Unsupported aggregation functions
    # aggregation = query_split[1].split("(")[0]
    # if aggregation.upper() not in ["SUM", "COUNT", "AVG", "VAR"]:
    #     continue

    # Export model name
    with open(out_filename, "a") as fp:
        fp.write(f"{aggregation_column},{list(set(predicate_columns))[0]}\n")
