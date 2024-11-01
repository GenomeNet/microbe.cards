import json
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Count, Prefetch
from jsonl_viewer.models import Microbe, PhenotypeDefinition, Prediction, PredictedPhenotype, PhenotypeSummary
from collections import defaultdict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Aggregates prediction summaries for all microbes and phenotypes.'
    
    BATCH_SIZE = 1000  # Adjust based on your system's memory

    def handle(self, *args, **options):
        self.stdout.write("Starting aggregation of phenotype summaries...")
        
        # Get total counts for progress bars
        total_microbes = Microbe.objects.count()
        phenotype_definitions = list(PhenotypeDefinition.objects.all())
        total_phenotypes = len(phenotype_definitions)
        
        self.stdout.write(f"Found {total_microbes} microbes and {total_phenotypes} phenotype definitions")

        with transaction.atomic():
            # Clear existing summaries
            PhenotypeSummary.objects.all().delete()
            self.stdout.write("Cleared existing PhenotypeSummary records.")

            # Process microbes in batches
            for offset in tqdm(range(0, total_microbes, self.BATCH_SIZE), desc="Processing batches", unit="batch"):
                summaries_to_create = []
                
                # Fetch microbes with prefetched predictions and phenotypes
                microbes_batch = (Microbe.objects
                    .select_related('taxonomy')
                    .prefetch_related(
                        Prefetch(
                            'predictions__predicted_phenotypes',
                            queryset=PredictedPhenotype.objects.select_related('definition')
                        )
                    )
                    .all()[offset:offset + self.BATCH_SIZE]
                )

                # Process each microbe in the batch
                for microbe in tqdm(microbes_batch, desc="Processing microbes in batch", leave=False):
                    # Create a dictionary to store predictions by phenotype definition
                    predictions_by_phenotype = defaultdict(list)
                    
                    # Collect all predictions for this microbe
                    for prediction in microbe.predictions.all():
                        for pred_phenotype in prediction.predicted_phenotypes.all():
                            if pred_phenotype.value not in [None, 'N/A', 'n/a']:
                                predictions_by_phenotype[pred_phenotype.definition_id].append(
                                    json.dumps(pred_phenotype.value).lower()
                                )

                    # Process each phenotype definition
                    for phenotype_def in phenotype_definitions:
                        values = predictions_by_phenotype.get(phenotype_def.id, [])
                        total_models = len(values)

                        if not values:
                            summaries_to_create.append(PhenotypeSummary(
                                microbe=microbe,
                                definition=phenotype_def,
                                majority_value='N/A',
                                agreement_percentage=0,
                                supporting_models=0,
                                total_models=total_models
                            ))
                            continue

                        # Count occurrences
                        value_counts = defaultdict(int)
                        for v in values:
                            value_counts[v] += 1

                        # Determine majority value
                        majority_value, majority_count = max(value_counts.items(), key=lambda x: x[1])

                        # Calculate agreement percentage
                        agreement_percentage = round((majority_count / len(values)) * 100)

                        summaries_to_create.append(PhenotypeSummary(
                            microbe=microbe,
                            definition=phenotype_def,
                            majority_value=json.loads(majority_value),
                            agreement_percentage=agreement_percentage,
                            supporting_models=majority_count,
                            total_models=total_models
                        ))

                # Bulk create the summaries for this batch
                PhenotypeSummary.objects.bulk_create(
                    summaries_to_create,
                    batch_size=1000
                )

        self.stdout.write(self.style.SUCCESS("\nAggregation of phenotype summaries completed successfully."))