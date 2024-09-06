import json
from django.core.management.base import BaseCommand
from django.db import transaction
from jsonl_viewer.models import Microbe, Taxonomy, Phenotype, PhenotypeDefinition
from tqdm import tqdm

class Command(BaseCommand):
    help = 'Import ground truth data and phenotypes from JSON files'

    def add_arguments(self, parser):
        parser.add_argument('ground_truth_file', type=str, help='Path to the ground truth JSON file')
        parser.add_argument('phenotypes_file', type=str, help='Path to the phenotypes JSON file')

    @transaction.atomic
    def handle(self, *args, **options):
        ground_truth_file = options['ground_truth_file']
        phenotypes_file = options['phenotypes_file']

        # Import phenotype definitions
        with open(phenotypes_file, 'r') as f:
            phenotypes_data = json.load(f)
        
        for name, data in phenotypes_data.items():
            PhenotypeDefinition.objects.update_or_create(
                name=name,
                defaults={
                    'data_type': data['dataType'],
                    'allowed_values': json.dumps(data.get('allowedValues', [])),
                    'description': data['description']
                }
            )

        # Import ground truth data
        with open(ground_truth_file, 'r') as f:
            ground_truth_data = json.load(f)

        total_entries = len(ground_truth_data['entries'])
        for entry in tqdm(ground_truth_data['entries'], total=total_entries, desc="Importing data"):
            taxonomy, _ = Taxonomy.objects.update_or_create(
                superkingdom=entry['taxonomy']['superkingdom'],
                phylum=entry['taxonomy']['phylum'],
                class_name=entry['taxonomy']['class'],
                order=entry['taxonomy']['order'],
                family=entry['taxonomy']['family'],
                genus=entry['taxonomy']['genus'],
                species=entry['taxonomy']['species']
            )

            microbe, _ = Microbe.objects.update_or_create(
                binomial_name=entry['binomialName'],
                defaults={
                    'ncbi_id': entry['ncbiId'],
                    'taxonomy': taxonomy,
                    'alternative_names': json.dumps(entry['alternativeNames']),
                    'ftp_path': entry['genomeInfo']['ftpPath'],
                    'fasta_file': entry['genomeInfo']['fastaFile']
                }
            )

            for phenotype_name, phenotype_data in entry['phenotypes'].items():
                phenotype_def = PhenotypeDefinition.objects.get(name=phenotype_name)
                if phenotype_def.data_type == 'boolean':
                    value = str(phenotype_data['value']).upper()
                else:
                    value = json.dumps(phenotype_data['value'])
                Phenotype.objects.update_or_create(
                    microbe=microbe,
                    definition=phenotype_def,
                    defaults={'value': value}
                )

        self.stdout.write(self.style.SUCCESS('Successfully imported ground truth data'))