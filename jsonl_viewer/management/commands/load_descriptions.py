import csv
import os
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from jsonl_viewer.models import Microbe, MicrobeDescription
from django.utils import timezone

class Command(BaseCommand):
    help = 'Load microbe descriptions from a CSV file into the database.'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The path to the CSV file to be imported.')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        if not os.path.isfile(csv_file):
            raise CommandError(f"File '{csv_file}' does not exist.")

        with open(csv_file, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                binomial_name = row.get('Binomial name')
                description = row.get('Description')
                model = row.get('Model')

                if not binomial_name or not description or not model:
                    self.stdout.write(self.style.WARNING(f"Skipping incomplete row: {row}"))
                    continue

                try:
                    microbe = Microbe.objects.get(binomial_name=binomial_name)
                except Microbe.DoesNotExist:
                    self.stdout.write(self.style.ERROR(f"Microbe with binomial name '{binomial_name}' does not exist. Skipping."))
                    continue

                # Ensure only one description per model type
                microbe_description, created = MicrobeDescription.objects.update_or_create(
                    microbe=microbe,
                    model=model,
                    defaults={
                        'description': description,
                        'inference_date_time': timezone.now()
                    }
                )

                if created:
                    self.stdout.write(self.style.SUCCESS(f"Added description for '{binomial_name}' by '{model}'."))
                else:
                    self.stdout.write(self.style.SUCCESS(f"Updated description for '{binomial_name}' by '{model}'."))

        self.stdout.write(self.style.SUCCESS('Import completed successfully.'))