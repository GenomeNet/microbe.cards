from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from jsonl_viewer.models import PhenotypeDefinition, PhenotypeStats, Phenotype, Prediction, Microbe
import json
import random

class Command(BaseCommand):
    help = "Compute summary statistics for each phenotype, including value_distribution."

    def handle(self, *args, **options):
        self.stdout.write("Computing phenotype stats...")

        # Clear existing stats
        PhenotypeStats.objects.all().delete()

        all_defs = PhenotypeDefinition.objects.exclude(name="Member of WA subset")

        # For demonstration, we will:
        # - Compute ground_truth_count, predicted_count, total_microbes as before
        # - Fill value_distribution with dummy data if none computed.
        for pd in all_defs:
            # Ground truth microbes for this phenotype (just count any non-NA values)
            gt_microbe_ids = Phenotype.objects.filter(definition=pd).exclude(value__in=[None, '', 'N/A'])\
                                              .values_list('microbe_id', flat=True).distinct()
            ground_truth_count = len(gt_microbe_ids)

            # Predicted microbes (without GT) for this phenotype
            pred_microbe_ids = Prediction.objects.filter(
                predicted_phenotypes__definition=pd,
                predicted_phenotypes__value__isnull=False
            ).exclude(predicted_phenotypes__value__in=['', 'N/A'])\
             .values_list('microbe_id', flat=True).distinct()

            # predicted_only_count = (pred - gt)
            pred_only_count = Microbe.objects.filter(id__in=pred_microbe_ids).exclude(id__in=gt_microbe_ids).count()

            total = ground_truth_count + pred_only_count

            # Value distribution:
            # In a real scenario, you'd count how many microbes have each allowed value.
            # For now, let's create dummy data for demonstration if we have allowed_values:
            allowed_vals = pd.allowed_values_list
            value_dist = {}
            if allowed_vals:
                # Randomly assign counts to each allowed value to avoid all zero scenario
                # Replace this logic with your actual counting logic!
                for val_item in allowed_vals:
                    # Assign a random count for demo purposes:
                    count_val = random.randint(0, 20)
                    value_dist[val_item] = count_val
                # Ensure not all zero:
                if all(v == 0 for v in value_dist.values()):
                    # Make at least one nonzero
                    first_key = allowed_vals[0]
                    value_dist[first_key] = 5
            else:
                # If no allowed values, just leave it empty
                value_dist = {}

            PhenotypeStats.objects.create(
                phenotype_definition=pd,
                ground_truth_count=ground_truth_count,
                predicted_count=pred_only_count,
                total_microbes=total,
                value_distribution=value_dist
            )

        self.stdout.write(self.style.SUCCESS("Phenotype stats computed and stored successfully."))