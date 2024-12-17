import os
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from jsonl_viewer.models import Microbe, Genome, Feature

class Command(BaseCommand):
    help = "Load GFF features from GFF files in a specified directory into the Feature model by matching fasta_file."

    def add_arguments(self, parser):
        parser.add_argument('--folder', type=str, required=True, help='Path to folder containing GFF files')
        parser.add_argument('--method_name', type=str, default='unknown_tool', help='Name of the method/tool that produced this GFF')
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
            self.stderr.write(self.style.WARNING("No .gff files found in the directory."))
            return

        # Print some sample entries from Microbe to see what we have
        self.stdout.write(self.style.NOTICE("Listing some Microbe fasta_file entries for debugging:"))
        sample_microbes = Microbe.objects.all()[:10]
        for m in sample_microbes:
            self.stdout.write(f"Microbe: {m.binomial_name}, fasta_file: {m.fasta_file}")

        for gff_file in gff_files:
            file_path = os.path.join(folder, gff_file)
            # Adjust the replacement to match your database naming scheme
            # Just print it first:
            self.stdout.write(self.style.NOTICE(f"Processing GFF file: {gff_file}"))

            # Example: remove '_genomic.gff' to match database entries if needed
            # If this doesn't match your DB entries, try different replacements
            #fasta_filename = gff_file.replace('_genomic.gff', '')
            fasta_filename = gff_file.replace('_genomic.gff', '_genomic.fasta')

            self.stdout.write(self.style.NOTICE(f"Derived fasta_filename: {fasta_filename}"))

            try:
                microbe = Microbe.objects.get(fasta_file=fasta_filename)
            except Microbe.DoesNotExist:
                self.stderr.write(self.style.ERROR(
                    f"No Microbe found matching fasta_file {fasta_filename} (derived from {gff_file})"))
                # Print possible similar entries:
                # For example, let's see if there's a close match by searching
                close_matches = Microbe.objects.filter(fasta_file__icontains=fasta_filename[:10])
                self.stdout.write("Potential similar fasta_file entries in DB:")
                for cm in close_matches[:5]:
                    self.stdout.write(f" - {cm.fasta_file}")
                continue

            try:
                genome = microbe.genome
            except Genome.DoesNotExist:
                self.stderr.write(self.style.ERROR(
                    f"No Genome found for microbe {microbe.binomial_name}. Please create a Genome entry first."))
                continue

            feature_id_map = {}
            now = timezone.now()

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

                        if 'ID' in attributes:
                            feature_id_map[attributes['ID']] = feat

            self.stdout.write(self.style.SUCCESS(
                f"Successfully loaded features from {gff_file} using {method_name} v{method_version} for {microbe.binomial_name}"))