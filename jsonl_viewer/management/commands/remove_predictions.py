from django.core.management.base import BaseCommand
from django.db import transaction
from jsonl_viewer.models import Microbe, Prediction, PhenotypeDefinition, PredictedPhenotype

class Command(BaseCommand):
    help = 'Remove all prediction data from the database'

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write("Removing all prediction data...")
        PredictedPhenotype.objects.all().delete()
        Prediction.objects.all().delete()
        PhenotypeDefinition.objects.all().delete()
        Microbe.objects.all().delete()
        self.stdout.write(self.style.SUCCESS("All prediction data has been removed."))