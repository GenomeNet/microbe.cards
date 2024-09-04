import csv
from django.core.management.base import BaseCommand
from jsonl_viewer.models import GroundTruthData


class Command(BaseCommand):
    help = 'Load ground truth data from CSV file into database'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to the CSV file')

    def handle(self, *args, **options):
        file_path = options['file_path']
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                GroundTruthData.objects.update_or_create(
                    binomial_name=row['Binomial name'],
                    defaults={
                        'ncbi_id': row['NCBI_ID'],
                        'motility': row['Motility'] == 'TRUE',
                        'gram_staining': row['Gram staining'],
                        'aerophilicity': row['Aerophilicity'] if row['Aerophilicity'] != 'NA' else None,
                        'extreme_environment_tolerance': row['Extreme environment tolerance'] == 'TRUE' if row['Extreme environment tolerance'] != 'NA' else None,
                        'biofilm_formation': row['Biofilm formation'] == 'TRUE' if row['Biofilm formation'] != 'NA' else None,
                        'animal_pathogenicity': row['Animal pathogenicity'] == 'TRUE' if row['Animal pathogenicity'] != 'NA' else None,
                        'biosafety_level': row['Biosafety level'],
                        'health_association': row['Health association'] == 'TRUE' if row['Health association'] != 'NA' else None,
                        'host_association': row['Host association'] == 'TRUE' if row['Host association'] != 'NA' else None,
                        'plant_pathogenicity': row['Plant pathogenicity'] == 'TRUE' if row['Plant pathogenicity'] != 'NA' else None,
                        'spore_formation': row['Spore formation'] == 'TRUE' if row['Spore formation'] != 'NA' else None,
                        'hemolysis': row['Hemolysis'] == 'TRUE' if row['Hemolysis'] != 'NA' else None,
                        'cell_shape': row['Cell shape'],
                        'member_of_wa_subset': row['Member of WA subset'] == 'TRUE',
                        'superkingdom': row['Superkingdom'],
                        'phylum': row['Phylum'],
                        'class_field': row['Class'],
                        'order': row['Order'],
                        'family': row['Family'],
                        'genus': row['Genus'],
                        'species': row['Species'],
                        'ftp_path': row['FTP path'] if row['FTP path'] != 'NA' else None,
                        'fasta_file': row['Fasta file'] if row['Fasta file'] != 'NA' else None,
                    }
                )
        self.stdout.write(self.style.SUCCESS('Successfully loaded ground truth data'))