from django.core.management.base import BaseCommand
from jsonl_viewer.models import PredictedPhenotype, PhenotypeDefinition, Microbe
from django.db.models import Count
from django.db.models.functions import TruncDate
import os

class Command(BaseCommand):
    help = 'Output statistics for gram staining phenotype and list binomial names for each group'

    def handle(self, *args, **options):
        # Find gram staining phenotype definition
        gram_stain_def = PhenotypeDefinition.objects.filter(name__icontains='gram').first()

        if not gram_stain_def:
            self.stdout.write(self.style.ERROR('No gram staining phenotype found.'))
            return

        self.stdout.write(self.style.SUCCESS(f'Using phenotype: {gram_stain_def.name}'))

        # Get value distribution
        value_distribution = PredictedPhenotype.objects.filter(
            definition=gram_stain_def
        ).exclude(
            value__in=['NA', 'null', '']
        ).values('value').annotate(count=Count('value')).order_by('-count')

        # Output statistics
        total_predictions = PredictedPhenotype.objects.filter(definition=gram_stain_def).count()
        valid_predictions = sum(item['count'] for item in value_distribution)

        self.stdout.write(self.style.SUCCESS(f'Total predictions: {total_predictions}'))
        self.stdout.write(self.style.SUCCESS(f'Valid (non-NA/non-missing) predictions: {valid_predictions}'))
        if total_predictions > 0:
            self.stdout.write(self.style.SUCCESS(f'Percentage of valid predictions: {valid_predictions/total_predictions*100:.2f}%'))

        if valid_predictions > 0:
            self.stdout.write(self.style.SUCCESS('\nValue distribution:'))
            for item in value_distribution:
                self.stdout.write(f"  {item['value']}: {item['count']} ({item['count']/valid_predictions*100:.2f}%)")

            # Create a directory for output files
            output_dir = 'gram_staining_groups'
            os.makedirs(output_dir, exist_ok=True)

            # Write binomial names for each group
            for item in value_distribution:
                group_value = item['value'].strip('"')  # Remove quotation marks
                safe_value = group_value.replace(' ', '_')  # Replace spaces with underscores
                filename = os.path.join(output_dir, f'gram_staining_{safe_value}.txt')
                
                binomial_names = PredictedPhenotype.objects.filter(
                    definition=gram_stain_def,
                    value=item['value']  # Use the original value for filtering
                ).values_list('prediction__microbe__binomial_name', flat=True).distinct()

                with open(filename, 'w') as f:
                    for name in binomial_names:
                        f.write(f"{name}\n")
                
                self.stdout.write(self.style.SUCCESS(f"Wrote {len(binomial_names)} binomial names to {filename}"))
        else:
            self.stdout.write(self.style.WARNING('No valid predictions found for this phenotype.'))

        # Get distribution by model
        model_distribution = PredictedPhenotype.objects.filter(
            definition=gram_stain_def
        ).exclude(
            value__in=['NA', 'null', '']
        ).values('prediction__model', 'value').annotate(count=Count('value')).order_by('prediction__model', '-count')

        self.stdout.write(self.style.SUCCESS('\nValue distribution by model:'))
        for item in model_distribution:
            self.stdout.write(f"  Model: {item['prediction__model']}, Value: {item['value']}, Count: {item['count']}")

        # Get distribution over time
        time_distribution = PredictedPhenotype.objects.filter(
            definition=gram_stain_def
        ).exclude(
            value__in=['NA', 'null', '']
        ).annotate(date=TruncDate('prediction__inference_date_time')).values('date', 'value').annotate(count=Count('value')).order_by('date', '-count')

        self.stdout.write(self.style.SUCCESS('\nValue distribution over time:'))
        for item in time_distribution:
            self.stdout.write(f"  Date: {item['date']}, Value: {item['value']}, Count: {item['count']}")
