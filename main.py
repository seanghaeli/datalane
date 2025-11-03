import os
import asyncio
import pandas as pd
import time
import csv
from typing import List
import sys
from loguru import logger

from src.models import BusinessRecord, MatchingResult
from src.search_query_set import expand_queries_for_record
from src.registry_fetcher import fetch_registry_for_record
from src.matchers.matching_orchestrator import matching_orchestrator
from src.config import INPUT_CSV, OUTPUT_CSV, BATCH_SIZE, LOG_LEVEL
from src.clients import ZyteClient

def load_businesses_from_csv(file_path: str, nrows: int = None) -> List[BusinessRecord]:
    """Load businesses from CSV and convert to BusinessRecord objects."""
    df = pd.read_csv(file_path, nrows=nrows)
    records = []
    for _, row in df.iterrows():
        # Helper to safely extract values from pandas Series, converting NaN to None
        def safe_get(col):
            if col not in row.index:
                return None
            val = row[col]
            if pd.isna(val):
                return None
            return val
        
        # Extract values with proper type conversion
        name = str(row["Name"]) if pd.notna(row["Name"]) else ""
        
        # Handle numeric conversions
        reviews_count = None
        if "Reviews count" in row.index and pd.notna(row["Reviews count"]):
            try:
                reviews_count = int(row["Reviews count"])
            except (ValueError, TypeError):
                reviews_count = None
        
        reviews_rating = None
        if "Reviews rating" in row.index and pd.notna(row["Reviews rating"]):
            try:
                reviews_rating = float(row["Reviews rating"])
            except (ValueError, TypeError):
                reviews_rating = None
        
        photos_count = None
        if "Photos count" in row.index and pd.notna(row["Photos count"]):
            photos_count = str(row["Photos count"])
        
        record = BusinessRecord(
            name=name,
            street_1=safe_get("Street 1"),
            description_1=safe_get("Description 1"),
            main_type=safe_get("Main type"),
            reviews_count=reviews_count,
            reviews_rating=reviews_rating,
            photos_count=photos_count,
        )
        records.append(record)
    return records

def batch_iter(records: List[BusinessRecord], batch_size: int):
    """
    Yield index and BusinessRecord slices of size `batch_size` for batched processing.
    """
    n = len(records)
    for i in range(0, n, batch_size):
        yield i, records[i:i+batch_size]

async def process_record(record: BusinessRecord) -> MatchingResult:
    """
    Process a single business record through the full matching pipeline.

    Args:
        record (BusinessRecord): Input business record.

    Returns:
        MatchingResult: Matching result for this business.
    """
    # 1) LLM query expansion
    expansion = await expand_queries_for_record(record)

    # 2) Fetch candidates from the government registry
    candidates = await fetch_registry_for_record(record, expansion)

    # 3) Determine whether government registry + Google Maps data support business existence
    result = await matching_orchestrator(record, candidates)

    return result

async def main():
    """
    Orchestrate the full batch processing pipeline.

    - Loads input CSV in batches.
    - Processes each batch asynchronously to determine if businesses should be kept.
    - Writes results incrementally to an output CSV.
    """
    # Load businesses from CSV
    all_businesses = load_businesses_from_csv(INPUT_CSV)
    
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

    # Process in batches, but use async.gather for parallelism within each batch
    try:
        for start_idx, batch_records in batch_iter(all_businesses, BATCH_SIZE):
            print(f"Processing rows {start_idx}..{start_idx + len(batch_records) - 1}")

            # Process all records in the batch in parallel
            results = await asyncio.gather(*[process_record(record) for record in batch_records])

            with open(output_path, "a", newline="") as f:
                writer = csv.writer(f)
                for result in results:
                    writer.writerow([
                        result.name,
                        result.results,
                        result.results_llm,
                        result.results_google_check,
                        result.overall_results,
                    ])
    finally:
        # Cleanup: close ZyteClient session to prevent unclosed connector warnings
        zyte_client = ZyteClient()
        await zyte_client.close()

if __name__ == "__main__":
    asyncio.run(main())