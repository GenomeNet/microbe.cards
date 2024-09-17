import json
from django.core.management.base import BaseCommand
from django.db import transaction, connection
from jsonl_viewer.models import Microbe, Prediction, PhenotypeDefinition, PredictedPhenotype
from django.utils.timezone import make_aware
from datetime import datetime
from tqdm import tqdm
from django.db.models import Count
import logging
from django.core.exceptions import ValidationError
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Import prediction data from JSON file'

    def add_arguments(self, parser):
        parser.add_argument('predictions_file', type=str, help='Path to the predictions JSON file')

    def get_table_counts(self):
        with connection.cursor() as cursor:
            counts = {}
            for table in ['jsonl_viewer_microbe', 'jsonl_viewer_prediction', 'jsonl_viewer_phenotypedefinition', 'jsonl_viewer_predictedphenotype']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
        return counts

    def is_valid_binomial_name(self, name):
        # Regular expression to match valid binomial names, including those with brackets
        pattern = r'^(\[?[A-Z][a-z]+\]?)\s([a-z]+)$'
        match = re.match(pattern, name)
        return bool(match)

    def handle(self, *args, **options):
        predictions_file = options['predictions_file']
        logger.info(f"Starting import from file: {predictions_file}")

        # Define batch_size
        batch_size = 1000  # You can adjust this value as needed

        # Get counts before import
        before_counts = self.get_table_counts()
        logger.info('Database state before import:')
        for table, count in before_counts.items():
            logger.info(f'  {table}: {count}')

        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)

        total_entries = len(predictions_data['predictions'])
        logger.info(f'Total entries in JSON file: {total_entries}')

        # Prepare bulk creation lists
        microbes_to_create = []
        predictions_to_create = []
        phenotypes_to_create = []
        phenotype_definitions = {}
        microbe_names = set()

        # Track statistics
        phenotype_value_counts = {}
        created_entries = 0
        error_count = 0

        # First, collect all unique microbes
        for entry in predictions_data['predictions']:
            microbe_names.add(entry['binomialName'])

        # Fetch existing microbes to avoid duplicates
        existing_microbes = Microbe.objects.filter(binomial_name__in=microbe_names).values_list('binomial_name', flat=True)
        existing_microbes_set = set(existing_microbes)

        # Prepare microbes to create
        for name in microbe_names:
            if not self.is_valid_binomial_name(name):
                logger.warning(f"Invalid binomial name skipped: {name}")
                continue
            if name not in existing_microbes_set:
                microbes_to_create.append(Microbe(binomial_name=name, ncbi_id=None))

        # Bulk create Microbes
        if microbes_to_create:
            Microbe.objects.bulk_create(microbes_to_create, batch_size=batch_size, ignore_conflicts=True)
            logger.info(f"Bulk created {len(microbes_to_create)} microbes.")

        # Create a mapping of binomial_name to Microbe instance
        microbe_map = {microbe.binomial_name: microbe for microbe in Microbe.objects.filter(binomial_name__in=microbe_names)}

        # Prepare Predictions and PredictedPhenotypes
        for entry in tqdm(predictions_data['predictions'], total=total_entries, desc="Processing predictions"):
            try:
                binomial_name = entry['binomialName']
                if not self.is_valid_binomial_name(binomial_name):
                    raise ValidationError(f"Invalid binomial name: {binomial_name}")

                microbe = microbe_map.get(binomial_name)
                if not microbe:
                    raise ValidationError(f"Microbe not found for binomial name: {binomial_name}")

                inference_date_time = make_aware(datetime.strptime(entry['inferenceDateTime'], "%Y-%m-%d %H:%M:%S"))

                prediction = Prediction(
                    microbe=microbe,
                    prediction_id=entry['predictionId'],
                    inference_date_time=inference_date_time,
                    model=entry['model']
                )
                predictions_to_create.append(prediction)

                for phenotype_name, phenotype_data in entry['phenotypes'].items():
                    if phenotype_name not in phenotype_definitions:
                        phen_def, created = PhenotypeDefinition.objects.get_or_create(name=phenotype_name)
                        phenotype_definitions[phenotype_name] = phen_def

                    phenotype = PredictedPhenotype(
                        definition=phenotype_definitions[phenotype_name],
                        value=json.dumps(phenotype_data['value'])
                    )
                    phenotypes_to_create.append(phenotype)

                    # Update the phenotype value counts
                    value = json.dumps(phenotype_data['value'])
                    if phenotype_name not in phenotype_value_counts:
                        phenotype_value_counts[phenotype_name] = {}
                    phenotype_value_counts[phenotype_name][value] = phenotype_value_counts[phenotype_name].get(value, 0) + 1

                created_entries += 1

            except Exception as e:
                error_count += 1
                logger.error(f"Error processing entry: {e}")
                logger.error(f"Entry data: {json.dumps(entry, indent=2)}")

        # Bulk create Predictions
        if predictions_to_create:
            Prediction.objects.bulk_create(predictions_to_create, batch_size=batch_size, ignore_conflicts=True)
            logger.info(f"Bulk created {len(predictions_to_create)} predictions.")

        # Refresh the Prediction instances to get their IDs
        prediction_ids = [pred.prediction_id for pred in predictions_to_create]
        prediction_map = {pred.prediction_id: pred for pred in Prediction.objects.filter(prediction_id__in=prediction_ids)}

        # Assign the correct Prediction instances to PredictedPhenotypes
        for phenotype in phenotypes_to_create:
            phenotype.prediction = prediction_map.get(phenotype.prediction.prediction_id)

        # Bulk create PredictedPhenotypes
        if phenotypes_to_create:
            PredictedPhenotype.objects.bulk_create(phenotypes_to_create, batch_size=batch_size, ignore_conflicts=True)
            logger.info(f"Bulk created {len(phenotypes_to_create)} predicted phenotypes.")

        logger.info('Successfully imported prediction data')

        # Get counts after import
        after_counts = self.get_table_counts()
        logger.info('Database state after import:')
        for table, count in after_counts.items():
            logger.info(f'  {table}: {count}')

        # Calculate and display differences
        logger.info('Entries added:')
        for table in before_counts.keys():
            diff = after_counts[table] - before_counts[table]
            logger.info(f'  {table}: {diff}')

        logger.info(f'Created entries: {created_entries}')
        logger.info(f'Entries with errors: {error_count}')

        # Additional statistics using raw SQL
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(DISTINCT microbe_id) FROM jsonl_viewer_prediction")
            unique_microbes = cursor.fetchone()[0]
            logger.info(f'Number of unique microbes with predictions: {unique_microbes}')

            cursor.execute("SELECT MAX(prediction_count) FROM (SELECT COUNT(*) as prediction_count FROM jsonl_viewer_prediction GROUP BY microbe_id) as subquery")
            max_predictions = cursor.fetchone()[0]
            logger.info(f'Maximum predictions for a single microbe: {max_predictions}')

            cursor.execute("SELECT COUNT(DISTINCT model) FROM jsonl_viewer_prediction")
            unique_models = cursor.fetchone()[0]
            logger.info(f'Number of unique models: {unique_models}')

        # Display phenotype counts and values
        logger.info('Predictions per phenotype and their values:')
        for phenotype, value_counts in phenotype_value_counts.items():
            logger.info(f'  {phenotype}:')
            total_count = sum(value_counts.values())
            for value, count in value_counts.items():
                logger.info(f'    {value}: {count} ({count/total_count:.2f}%)')

        # Validate database counts for each phenotype
        logger.info('Database counts per phenotype:')
        db_phenotype_counts = PredictedPhenotype.objects.values('definition__name', 'value').annotate(count=Count('id'))
        for item in db_phenotype_counts:
            phenotype_name = item['definition__name']
            value = item['value']
            count = item['count']
            logger.info(f'  {phenotype_name} - {value}: {count}')

        # Summary
        logger.info(f"Total predictions processed: {total_entries}")
        logger.info(f"Successful imports: {created_entries}")
        logger.info(f"Errors during processing: {error_count}")