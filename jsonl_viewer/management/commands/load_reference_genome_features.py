import os
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from jsonl_viewer.models import Microbe, Genome, Feature  # Adjust 'yourapp' to your actual app name

class Command(BaseCommand):
    help = "Load genome features from GFF files into the database for the reference genome of each Microbe."

    def add_arguments(self, parser):
        parser.add_argument('--folder', type=str, required=True, help='Path to folder containing GFF files')
        parser.add_argument('--method_name', type=str, default='unknown_tool', help='Name of the method/tool that produced these annotations')
        parser.add_argument('--method_version', type=str, default='unknown_version', help='Version of the method/tool')

    def handle(self, *args, **options):
        folder = options['folder']
        method_name = options['method_name']
        method_version = options['method_version']

        if not os.path.isdir(folder):
            self.stderr.write(self.style.ERROR(f"{folder} is not a directory"))
            return

        gff_files = [f for f in os.listdir(folder) if f.endswith('.gff')]
        if not gff_files:
            self.stderr.write(self.style.WARNING("No .gff files found in the specified directory."))
            return

        now = timezone.now()

        for gff_file in gff_files:
            file_path = os.path.join(folder, gff_file)
            # Derive the corresponding fasta filename (adjust this logic if your naming differs)
            fasta_filename = gff_file.replace('_genomic.gff', '_genomic.fasta')

            try:
                microbe = Microbe.objects.get(fasta_file=fasta_filename)
            except Microbe.DoesNotExist:
                self.stderr.write(self.style.ERROR(
                    f"No Microbe found for fasta_file '{fasta_filename}' derived from '{gff_file}'. Check naming conventions."))
                continue

            # Ensure a Genome object exists for this Microbe
            if not hasattr(microbe, 'genome'):
                # Create a Genome entry if it doesn't exist. 
                # Use the fasta_filename as a placeholder for assembly_accession if not readily available.
                genome = Genome.objects.create(
                    microbe=microbe,
                    assembly_accession=fasta_filename,
                )
            else:
                genome = microbe.genome

            self.stdout.write(self.style.NOTICE(f"Loading features from '{gff_file}' into genome '{genome.assembly_accession}'."))

            feature_id_map = {}
            with open(file_path, 'r') as f:
                with transaction.atomic():
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        parts = line.split('\t')
                        if len(parts) < 9:
                            continue

                        seqid, source, ftype, start, end, score, strand, phase, attributes_str = parts

                        # Parse the attributes into a dictionary
                        attributes = {}
                        for attr_pair in attributes_str.split(';'):
                            attr_pair = attr_pair.strip()
                            if '=' in attr_pair:
                                key, val = attr_pair.split('=', 1)
                                attributes[key] = val

                        start = int(start)
                        end = int(end)

                        parent_id = attributes.get('Parent')
                        parent_feature = feature_id_map.get(parent_id)

                        methods_dict = {
                            method_name: {
                                "version": method_version,
                                "date_imported": now.isoformat()
                            }
                        }

                        feat = Feature.objects.create(
                            genome=genome,
                            feature_type=ftype,
                            seqid=seqid,
                            source=source,
                            start=start,
                            end=end,
                            strand=strand if strand in ['+', '-'] else None,
                            attributes=attributes,
                            parent=parent_feature,
                            methods=methods_dict,
                            imported_at=now
                        )

                        # If the feature has an ID, store it in the map for parent-child linking
                        if 'ID' in attributes:
                            feature_id_map[attributes['ID']] = feat

            self.stdout.write(self.style.SUCCESS(
                f"Successfully loaded features from '{gff_file}' for microbe '{microbe.binomial_name}'."))