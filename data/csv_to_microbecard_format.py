import csv
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Set

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_boolean(value: str) -> bool | None:
    if value.upper() == 'TRUE':
        return True
    elif value.upper() == 'FALSE':
        return False
    return None

def get_data_type(values: Set[str]) -> str:
    if all(val.upper() in ['TRUE', 'FALSE', 'NA'] for val in values):
        return 'boolean'
    try:
        float(next(val for val in values if val != 'NA'))
        return 'numerical'
    except ValueError:
        return 'categorical'

def validate_ncbi_id(ncbi_id: str) -> int | None:
    try:
        return int(ncbi_id)
    except ValueError:
        logging.warning(f"Invalid NCBI ID: {ncbi_id}")
        return None

def validate_taxonomy(taxonomy: Dict[str, str]) -> List[str]:
    required_fields = ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]
    missing_fields = [field for field in required_fields if not taxonomy.get(field)]
    return missing_fields

def process_row(row: Dict[str, str], phenotype_definitions: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "binomialName": row["Binomial name"],
        "ncbiId": validate_ncbi_id(row["NCBI_ID"]),
        "taxonomy": {
            "superkingdom": row["Superkingdom"],
            "phylum": row["Phylum"],
            "class": row["Class"],
            "order": row["Order"],
            "family": row["Family"],
            "genus": row["Genus"],
            "species": row["Species"]
        },
        "alternativeNames": [],
        "phenotypes": {},
        "genomeInfo": {
            "ftpPath": row["FTP path"] if row["FTP path"] != "NA" else None,
            "fastaFile": row["Fasta file"] if row["Fasta file"] != "NA" else None
        }
    }

    missing_taxonomy = validate_taxonomy(entry["taxonomy"])
    if missing_taxonomy:
        logging.warning(f"Missing taxonomy fields for {entry['binomialName']}: {', '.join(missing_taxonomy)}")

    for phenotype, definition in phenotype_definitions.items():
        value = row.get(phenotype)
        if value and value != "NA":
            if definition["dataType"] == "boolean":
                value = parse_boolean(value)
            elif definition["dataType"] == "numerical":
                try:
                    value = float(value)
                except ValueError:
                    logging.warning(f"Invalid numerical value '{value}' for phenotype '{phenotype}' in {entry['binomialName']}")
                    continue
            
            entry["phenotypes"][phenotype] = {
                "value": value
            }

    return entry

def analyze_phenotypes(csv_file: str) -> Dict[str, Any]:
    phenotypes = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames
        for header in headers:
            if header not in ["Binomial name", "NCBI_ID", "Superkingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species", "FTP path", "Fasta file"]:
                phenotypes[header] = {"values": set()}
        
        for row in reader:
            for phenotype in phenotypes:
                value = row.get(phenotype)
                if value and value != "NA":
                    phenotypes[phenotype]["values"].add(value)
    
    for phenotype, data in phenotypes.items():
        data["dataType"] = get_data_type(data["values"])
        if data["dataType"] == "categorical":
            data["allowedValues"] = list(data["values"])
        del data["values"]
        data["description"] = f"Description for {phenotype}"  # Placeholder, to be filled manually

    return phenotypes

def main(csv_file: str, data_source: str, output_file: str, phenotype_spec_file: str, verbose: bool) -> None:
    setup_logging(verbose)
    phenotype_definitions = analyze_phenotypes(csv_file)
    entries = []
    processed_count = 0
    error_count = 0

    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    entry = process_row(row, phenotype_definitions)
                    if entry["ncbiId"] is None:
                        raise ValueError(f"Invalid NCBI ID for {entry['binomialName']}")
                    entries.append(entry)
                    logging.info(f"Processed: {entry['binomialName']}")
                    processed_count += 1
                except Exception as e:
                    logging.error(f"Error processing row: {row['Binomial name']} - {str(e)}")
                    error_count += 1

        output_data = {
            "entries": entries,
            "dataSource": data_source,
            "lastUpdated": datetime.now().isoformat()
        }

        with open(output_file, 'w') as out_file:
            json.dump(output_data, out_file, indent=2)

        with open(phenotype_spec_file, 'w') as spec_file:
            json.dump(phenotype_definitions, spec_file, indent=2)

    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file}")
        return

    logging.info(f"Processing complete. {processed_count} entries processed, {error_count} errors encountered.")
    logging.info(f"Ground truth data written to {output_file}")
    logging.info(f"Phenotype specifications written to {phenotype_spec_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to microbe.card Ground Truth Data format")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument("--data_source", required=True, help="Primary source of the ground truth data")
    parser.add_argument("--output_file", default="microbe_card_ground_truth.json", help="Output JSON file for ground truth data")
    parser.add_argument("--phenotype_spec_file", default="phenotype_specifications.json", help="Output JSON file for phenotype specifications")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    main(args.csv_file, args.data_source, args.output_file, args.phenotype_spec_file, args.verbose)