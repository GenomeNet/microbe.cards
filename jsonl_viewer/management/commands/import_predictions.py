import json
from django.core.management.base import BaseCommand
from django.db import transaction
from jsonl_viewer.models import Microbe, Prediction, PhenotypeDefinition, PredictedPhenotype
from django.utils.timezone import make_aware
from datetime import datetime
from tqdm import tqdm

class Command(BaseCommand):
    help = 'Import prediction data from JSON file'

    def add_arguments(self, parser):
        parser.add_argument('predictions_file', type=str, help='Path to the predictions JSON file')

    @transaction.atomic
    def handle(self, *args, **options):
        predictions_file = options['predictions_file']

        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)

        total_entries = len(predictions_data['predictions'])
        for entry in tqdm(predictions_data['predictions'], total=total_entries, desc="Importing predictions"):
            microbe, _ = Microbe.objects.get_or_create(binomial_name=entry['binomialName'])

            # Convert the string datetime to a timezone-aware datetime object
            inference_date_time = make_aware(datetime.strptime(entry['inferenceDateTime'], "%Y-%m-%d %H:%M:%S"))

            prediction, created = Prediction.objects.update_or_create(
                microbe=microbe,
                prediction_id=entry['predictionId'],
                inference_date_time=inference_date_time,
                defaults={'model': entry['model']}
            )

            for phenotype_name, phenotype_data in entry['phenotypes'].items():
                phenotype_def, _ = PhenotypeDefinition.objects.get_or_create(name=phenotype_name)
                PredictedPhenotype.objects.update_or_create(
                    prediction=prediction,
                    definition=phenotype_def,
                    defaults={'value': json.dumps(phenotype_data['value'])}
                )

        self.stdout.write(self.style.SUCCESS('Successfully imported prediction data'))