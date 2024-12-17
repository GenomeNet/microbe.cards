import os
import json
import datetime
import logging
import time  # Add this line
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db import IntegrityError, transaction
from jsonl_viewer.models import Microbe, Manuscript


# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('import_manuscripts.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Command(BaseCommand):
    help = "Step 2: Import manuscripts from JSON files downloaded by download_manuscripts_json."

    def add_arguments(self, parser):
        parser.add_argument(
            '--input-dir',
            default='data/manuscripts_raw',
            help='Directory containing the JSON files.'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run without saving to the database.'
        )

    def handle(self, *args, **options):
        input_dir = options['input_dir']
        dry_run = options['dry_run']

        start_time = time.time()

        files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        self.stdout.write(self.style.NOTICE(f"Found {len(files)} JSON files in {input_dir}."))
        logger.info(f"Found {len(files)} JSON files in {input_dir}.")

        total_imported = 0
        processed_files = 0

        for filename in files:
            filepath = os.path.join(input_dir, filename)
            microbe_name = filename.replace('_', ' ').replace('.json', '')

            # Try to find the microbe in DB
            try:
                microbe = Microbe.objects.get(binomial_name=microbe_name)
            except Microbe.DoesNotExist:
                self.stdout.write(self.style.WARNING(f"Microbe {microbe_name} not found in DB, skipping file {filename}"))
                logger.warning(f"Microbe {microbe_name} not found in DB, skipping file {filename}")
                continue

            self.stdout.write(self.style.NOTICE(f"Importing manuscripts for {microbe_name} from {filename}"))
            logger.info(f"Importing manuscripts for {microbe_name} from {filename}")

            with open(filepath, 'r', encoding='utf-8') as f:
                articles = json.load(f)

            imported_count = 0
            for art in articles:
                title = art['title']
                if not title:
                    continue

                # Check duplicate
                if Manuscript.objects.filter(microbe=microbe, title=title).exists():
                    logger.debug(f"Duplicate found: {title[:60]}... for {microbe_name}")
                    continue

                # Convert published_date back to date object if present
                pub_date_str = art.get('published_date')
                pub_date = None
                if pub_date_str:
                    try:
                        pub_date = datetime.date.fromisoformat(pub_date_str)
                    except ValueError:
                        pass

                manuscript_data = {
                    'microbe': microbe,
                    'title': title,
                    'abstract': art.get('abstract', ''),
                    'authors': art.get('authors', []),
                    'pmcid': art.get('pmcid'),
                    'source': art.get('source', ''),
                    'journal_title': art.get('journal_title', ''),
                    'is_open_access': art.get('is_open_access', False),
                    'published_year': art.get('published_year'),
                    'published_date': pub_date
                }

                if dry_run:
                    logger.info(f"[DRY RUN] Would import: {title[:60]}...")
                    imported_count += 1
                else:
                    try:
                        with transaction.atomic():
                            Manuscript.objects.create(**manuscript_data)
                        imported_count += 1
                        self.stdout.write(self.style.SUCCESS(f"Imported: {title[:60]}..."))
                        logger.info(f"Imported: {title[:60]}... (Microbe: {microbe_name})")
                    except IntegrityError:
                        self.stdout.write(self.style.WARNING("Duplicate or error on insert, skipping."))
                        logger.warning(f"Duplicate or error on insert for {title[:60]}...")

            self.stdout.write(self.style.NOTICE(f"Done importing {imported_count} manuscripts for {microbe_name}"))
            logger.info(f"Done importing {imported_count} manuscripts for {microbe_name}")

            total_imported += imported_count
            processed_files += 1

        elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f"Completed import from {processed_files} files. Total imported: {total_imported}. Elapsed: {elapsed:.2f}s"
        ))
        logger.info(f"Completed import from {processed_files} files. Total imported: {total_imported}. Elapsed: {elapsed:.2f}s")
