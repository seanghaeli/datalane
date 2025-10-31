import os
import asyncio
import pandas as pd
import time
import csv
from typing import List
import sys
from loguru import logger

from src.search_query_set import expand_queries_for_batch
from src.registry_fetcher import fetch_registry_for_batch
from src.matchers.matching_orchestrator import matching_orchestrator
from src.config import INPUT_CSV, OUTPUT_CSV, BATCH_SIZE, LOG_LEVEL

def batch_iter(df: pd.DataFrame, batch_size: int):
    """
    Yield index and DataFrame slices of size `batch_size` for batched processing.
    """
    n = len(df)
    for i in range(0, n, batch_size):
        yield i, df.iloc[i:i+batch_size].copy()

async def process_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a single batch of business records through the full matching pipeline.

    Args:
        batch_df (pd.DataFrame): Batch of input rows containing business names,
                                 addresses, and other metadata.

    Returns:
        pd.DataFrame: Columns ["Name", "results", "resultsLLM", "overallResults"].
    """
    start = time.perf_counter()

    # 1) LLM query expansion (single call for the whole batch)
    search_queries = await expand_queries_for_batch(batch_df)

    # 2) Fetch candidates from the government registry
    candidates = await fetch_registry_for_batch(batch_df, search_queries)

    # # 3) Determine whether government registry + Google Maps data support business existence
    results_df = await matching_orchestrator(batch_df, candidates)

    elapsed = time.perf_counter() - start
    logger.debug(f"Batch of {len(batch_df)} rows processed in {elapsed:.2f} seconds")
    return results_df

async def main():
    """
    Orchestrate the full batch processing pipeline.

    - Loads input CSV in batches.
    - Processes each batch asynchronously to determine if businesses should be kept.
    - Writes results incrementally to an output CSV.
    """
    # df = pd.read_csv(INPUT_CSV, skiprows=range(1, 160), nrows=40)
    df = pd.read_csv(INPUT_CSV, nrows=160)
    # Initialize logs
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level=LOG_LEVEL, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    # Initialize output file
    output_path = OUTPUT_CSV
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "results", "resultsLLM", "resultsGoogleCheck", "overallResults"])

    # Process each batch sequentially
    for start_idx, batch_df in batch_iter(df, BATCH_SIZE):
        print(f"Processing rows {start_idx}..{start_idx + len(batch_df) - 1}")

        results = await process_batch(batch_df)

        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            for _, row in results.iterrows():
                writer.writerow([
                    row.get("Name"),
                    bool(row.get("results", False)),
                    bool(row.get("resultsLLM", False)),
                    int(row.get("resultsGoogleCheck", 0)),
                    bool(row.get("overallResults", False)),
                ])

if __name__ == "__main__":
    asyncio.run(main())