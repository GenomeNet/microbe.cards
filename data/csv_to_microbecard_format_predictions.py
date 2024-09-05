import csv
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List
import uuid

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_column_range(column_range: str) -> List[int]:
    columns = []
    for part in column_range.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            columns.extend(range(start, end + 1))
        else:
            columns.append(int(part))
    return columns

def process_row(row: List[str], prediction_id: str, headers: List[str], prediction_indices: List[int], binomial_index: int) -> Dict[str, Any]:
    entry = {
        "binomialName": row[binomial_index],
        "phenotypes": {},
        "model": row[headers.index("Model")],
        "inferenceDateTime": row[headers.index("Infrence date and time")],
        "predictionId": prediction_id
    }

    for i in prediction_indices:
        phenotype = headers[i]
        value = row[i]
        if value and value != "NA":
            entry["phenotypes"][phenotype] = {"value": value}

    return entry

def main(csv_file: str, output_file: str, prediction_columns: str, binomial_column: int, verbose: bool) -> None:
    setup_logging(verbose)
    entries = []
    prediction_id = str(uuid.uuid4())
    processed_count = 0
    error_count = 0
    model_name = None

    prediction_indices = parse_column_range(prediction_columns)
    binomial_index = binomial_column - 1  # Convert to 0-based index

    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            for row in reader:
                try:
                    entry = process_row(row, prediction_id, headers, prediction_indices, binomial_index)
                    entries.append(entry)
                    logging.info(f"Processed: {entry['binomialName']}")
                    processed_count += 1
                    
                    # Set model_name from the first row (assuming it's consistent across all rows)
                    if model_name is None:
                        model_name = entry['model']
                except Exception as e:
                    logging.error(f"Error processing row: {row[binomial_index]} - {str(e)}")
                    error_count += 1

        output_data = {
            "predictions": entries,
            "modelName": model_name,
            "predictionId": prediction_id,
            "lastUpdated": datetime.now().isoformat()
        }

        with open(output_file, 'w') as out_file:
            json.dump(output_data, out_file, indent=2)

    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file}")
        return

    logging.info(f"Processing complete. {processed_count} entries processed, {error_count} errors encountered.")
    logging.info(f"Prediction data written to {output_file}")
    logging.info(f"Model used: {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to Prediction Data JSON format")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument("--output_file", default="prediction_data.json", help="Output JSON file for prediction data")
    parser.add_argument("--prediction_columns", required=True, help="Comma-separated list or range of column indices for predictions (e.g., '2-5,7,9-11')")
    parser.add_argument("--binomial_column", type=int, required=True, help="Column index for binomial names (1-based)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    main(args.csv_file, args.output_file, args.prediction_columns, args.binomial_column, args.verbose)