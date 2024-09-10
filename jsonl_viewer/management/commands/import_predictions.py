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
        microbe_dict = {}  # New dictionary to store created microbes

        # Add dictionaries to keep track of various statistics
        phenotype_value_counts = {}
        created_entries = 0
        error_count = 0

        for entry in tqdm(predictions_data['predictions'], total=total_entries, desc="Processing predictions"):
            try:
                if not self.is_valid_binomial_name(entry['binomialName']):
                    raise ValidationError(f"Invalid binomial name: {entry['binomialName']}")

                microbe = Microbe(
                    binomial_name=entry['binomialName'],
                    ncbi_id=None  # We don't have this information in the JSON
                )
                microbes_to_create.append(microbe)
                microbe_dict[entry['binomialName']] = microbe

                inference_date_time = make_aware(datetime.strptime(entry['inferenceDateTime'], "%Y-%m-%d %H:%M:%S"))

                prediction = Prediction(
                    microbe=microbe,
                    prediction_id=entry['predictionId'],
                    inference_date_time=inference_date_time,
                    model=entry['model']
                )
                predictions_to_create.append(prediction)

                for phenotype_name, phenotype_data in entry['phenotypes'].items():
                    phenotype = PredictedPhenotype(
                        prediction=prediction,
                        definition=PhenotypeDefinition.objects.get_or_create(name=phenotype_name)[0],
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

        # Bulk create Microbes
        with transaction.atomic():
            Microbe.objects.bulk_create(microbes_to_create, batch_size=batch_size, ignore_conflicts=True)

        # Now create Prediction objects
        for prediction in predictions_to_create:
            prediction.microbe = Microbe.objects.get(binomial_name=prediction.microbe.binomial_name)

        # Bulk create Predictions
        with transaction.atomic():
            for prediction in predictions_to_create:
                Prediction.objects.update_or_create(
                    microbe=prediction.microbe,
                    prediction_id=prediction.prediction_id,
                    inference_date_time=prediction.inference_date_time,
                    model=prediction.model,
                    defaults={} # Add any other fields here if needed
                )

        # Remove the bulk creation of PredictedPhenotypes and replace with:
        for phenotype in phenotypes_to_create:
            prediction, _ = Prediction.objects.get_or_create(
                microbe=phenotype.prediction.microbe,
                prediction_id=phenotype.prediction.prediction_id,
                inference_date_time=phenotype.prediction.inference_date_time,
                model=phenotype.prediction.model
            )
            PredictedPhenotype.objects.update_or_create(
                prediction=prediction,
                definition=phenotype.definition,
                defaults={'value': phenotype.value}
            )

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