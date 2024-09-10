from django.core.management.base import BaseCommand
from django.db.models import Count
from jsonl_viewer.models import PhenotypeDefinition, Phenotype, PredictedPhenotype, Prediction
import pandas as pd

class Command(BaseCommand):
    help = 'Generate summary statistics for phenotypes and predictions'

    def handle(self, *args, **options):
        self.stdout.write("Generating summary statistics...")

        statistics = []

        phenotypes = PhenotypeDefinition.objects.exclude(name="Member of WA subset")

        for phenotype in phenotypes:
            self.stdout.write(f"Processing phenotype: {phenotype.name}")

            ground_truth_count = Phenotype.objects.filter(definition=phenotype).count()
            ground_truth_unique_microbes = Phenotype.objects.filter(definition=phenotype).values('microbe').distinct().count()

            predictions_count = PredictedPhenotype.objects.filter(definition=phenotype).count()
            predictions_unique_microbes = PredictedPhenotype.objects.filter(definition=phenotype).values('prediction__microbe').distinct().count()

            unique_models = PredictedPhenotype.objects.filter(definition=phenotype).values('prediction__model').distinct().count()

            statistics.append({
                'Phenotype': phenotype.name,
                'GroundTruthCount': ground_truth_count,
                'GroundTruthUniqueMicrobes': ground_truth_unique_microbes,
                'PredictionsCount': predictions_count,
                'PredictionsUniqueMicrobes': predictions_unique_microbes,
                'UniqueModels': unique_models,
            })

        # Convert to DataFrame for easy viewing and saving
        df = pd.DataFrame(statistics)

        # Add total rows
        total_row = df.sum(numeric_only=True).to_dict()
        total_row['Phenotype'] = 'TOTAL'
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        # Print the statistics
        print(df.to_string(index=False))

        # Save to CSV
        df.to_csv('phenotype_statistics.csv', index=False)

        self.stdout.write(self.style.SUCCESS('Summary statistics generated successfully'))

        # Additional overall statistics
        total_unique_microbes = Phenotype.objects.exclude(definition__name="Member of WA subset").values('microbe').distinct().count()
        total_unique_predicted_microbes = Prediction.objects.filter(predicted_phenotypes__definition__name__ne="Member of WA subset").values('microbe').distinct().count()

        self.stdout.write(f"\nTotal unique microbes in ground truth: {total_unique_microbes}")
        self.stdout.write(f"Total unique microbes in predictions: {total_unique_predicted_microbes}")
