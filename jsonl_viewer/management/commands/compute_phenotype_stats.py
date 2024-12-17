from django.core.management.base import BaseCommand
from jsonl_viewer.models import PhenotypeDefinition, Phenotype, Prediction, PhenotypeStats, Microbe
from django.db.models import Count

class Command(BaseCommand):
    help = "Compute summary statistics for each phenotype and store in PhenotypeStats."

    def handle(self, *args, **options):
        self.stdout.write("Computing phenotype stats...")
        
        # Clear existing stats
        PhenotypeStats.objects.all().delete()

        # Get all phenotypes
        all_defs = PhenotypeDefinition.objects.exclude(name="Member of WA subset")

        # We'll compute:
        # ground_truth_count: microbes that have a ground_truth value for this phenotype (non-null, non-empty, != 'N/A')
        # predicted_count: microbes that have no ground_truth but have at least one predicted value
        # total_microbes: union of ground_truth_count and predicted_count
        # This might require multiple queries or a more complex approach.

        # Get all microbe IDs that have ground truth for a given phenotype
        # A ground truth phenotype is one where Phenotype.value is not null or 'N/A'
        ground_truth_map = {}
        for pd in all_defs:
            q = Phenotype.objects.filter(definition=pd).exclude(value__in=[None, '', 'N/A'])
            ground_truth_count = q.values('microbe').distinct().count()
            ground_truth_map[pd.id] = ground_truth_count

        # For predicted count, we find microbes that do not have ground truth but do have predictions
        # PredictedPhenotype is linked via Prediction, so we check if there's at least one predicted value
        # not in ground truth.
        predicted_map = {}
        for pd in all_defs:
            # Microbes with ground truth for this pd
            gt_microbe_ids = Phenotype.objects.filter(definition=pd).exclude(value__in=[None, '', 'N/A']).values_list('microbe_id', flat=True)

            # Microbes with predicted phenotypes for this pd
            # We'll find any microbe that has a predicted_phenotype for pd
            # and exclude those that already have ground truth.
            predicted_microbe_ids = Prediction.objects.filter(
                predicted_phenotypes__definition=pd,
                predicted_phenotypes__value__isnull=False
            ).exclude(predicted_phenotypes__value__in=['', 'N/A']).values_list('microbe_id', flat=True).distinct()

            # Predicted only = predicted - ground truth
            predicted_only_count = Microbe.objects.filter(id__in=predicted_microbe_ids).exclude(id__in=gt_microbe_ids).count()
            predicted_map[pd.id] = predicted_only_count

        # Total = ground_truth_count + predicted_count
        for pd in all_defs:
            gt_count = ground_truth_map[pd.id]
            pred_count = predicted_map[pd.id]
            total = gt_count + pred_count

            PhenotypeStats.objects.create(
                phenotype_definition=pd,
                ground_truth_count=gt_count,
                predicted_count=pred_count,
                total_microbes=total
            )

        self.stdout.write(self.style.SUCCESS("Phenotype stats computed and stored successfully."))