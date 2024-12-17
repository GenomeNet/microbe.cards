import os
import time
import requests
import json
import datetime
import logging
from django.core.management.base import BaseCommand
from django.conf import settings
from jsonl_viewer.models import Microbe

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('download_manuscripts_json.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PAGE_SIZE = 100000
MAX_RESULTS_PER_SPECIES = 500000
MAX_RETRIES = 3

def robust_request(url, params, timeout=30, max_retries=MAX_RETRIES):
    attempt = 0
    while attempt < max_retries:
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            attempt += 1
            sleep_time = 2 ** attempt
            logger.warning(f"Request error ({e}), attempt {attempt}/{max_retries}, sleeping {sleep_time}s...")
            time.sleep(sleep_time)
    raise RuntimeError("Max retries reached for request. Could not complete the request.")

def search_europe_pmc(query, page=1, pageSize=PAGE_SIZE):
    params = {
        'query': query,
        'format': 'json',
        'page': page,
        'pageSize': pageSize,
        'resultType': 'core'  
    }
    response = robust_request(f"{BASE_URL}/search", params=params)
    return response.json()

def parse_date(article):
    pub_year = article.get('pubYear', None)
    pub_date_str = article.get('firstPublicationDate', '')
    pub_date = None
    if pub_date_str:
        try:
            pub_date = datetime.date.fromisoformat(pub_date_str)
        except ValueError:
            pass
    return pub_year, pub_date

class Command(BaseCommand):
    help = (
        "Step 1: Download manuscripts/abstracts from Europe PMC for all microbes "
        "and save them as JSON files per microbe for later import."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--output-dir',
            default='data/manuscripts_raw',
            help='Directory to store the downloaded JSON files.'
        )
        parser.add_argument(
            '--max-results',
            type=int,
            default=MAX_RESULTS_PER_SPECIES,
            help='Maximum number of manuscripts to download per species.'
        )

    def handle(self, *args, **options):
        output_dir = options['output_dir']
        max_results = options['max_results']
        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        microbes = Microbe.objects.all().order_by('id')
        species_count = microbes.count()
        self.stdout.write(self.style.NOTICE(f"Found {species_count} microbes. Starting download..."))
        logger.info(f"Found {species_count} microbes. Starting download...")

        total_collected = 0
        processed_species = 0

        for microbe in microbes:
            microbe_name = microbe.binomial_name.replace(' ', '_')
            json_path = os.path.join(output_dir, f"{microbe_name}.json")

            # Skip if already exists
            if os.path.exists(json_path):
                self.stdout.write(self.style.WARNING(f"{microbe.binomial_name} JSON already exists, skipping download."))
                logger.info(f"{microbe.binomial_name} JSON already exists, skipping download.")
                processed_species += 1
                continue

            self.stdout.write(self.style.NOTICE(f"Downloading for {microbe.binomial_name}"))
            logger.info(f"Downloading for {microbe.binomial_name}")

            articles = self.fetch_manuscripts(microbe, max_results=max_results)
            total_collected += len(articles)

            # Save to JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2)
            self.stdout.write(self.style.SUCCESS(f"Saved {len(articles)} manuscripts for {microbe.binomial_name} to {json_path}"))
            logger.info(f"Saved {len(articles)} manuscripts for {microbe.binomial_name} to {json_path}")

            processed_species += 1

        elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f"Completed downloading for {processed_species} microbes. Total articles: {total_collected}. Elapsed: {elapsed:.2f}s"
        ))
        logger.info(f"Completed downloading for {processed_species} microbes. Total articles: {total_collected}. Elapsed: {elapsed:.2f}s")

    def fetch_manuscripts(self, microbe, max_results=MAX_RESULTS_PER_SPECIES):
        query = microbe.binomial_name
        page = 1
        collected = 0
        all_articles = []
        
        # Added line ~172
        seen_pmids = set()  # Track unique PMCID to avoid duplicates

        while True:
            try:
                data = search_europe_pmc(query, page=page, pageSize=PAGE_SIZE)
            except RuntimeError as e:
                self.stdout.write(self.style.ERROR(f"Error querying {query}: {e}"))
                logger.error(f"Error querying {query}: {e}")
                break

            results = data.get('resultList', {}).get('result', [])
            hit_count = int(data.get('hitCount', 0))

            if page == 1:
                logger.info(f"{query}: Found {hit_count} total results.")
                self.stdout.write(self.style.NOTICE(f"{query}: Found {hit_count} total results."))

            if not results:
                # No more results
                break

            before_add_count = len(all_articles)
            for article in results:
                if collected >= max_results:
                    break

                title = (article.get('title') or '').strip()
                abstract = (article.get('abstractText') or '').strip()
                authors_data = article.get('authorList', {}).get('author', [])
                authors = [a.get('fullName', '').strip() for a in authors_data if a.get('fullName')]
                pmcid = article.get('pmcid', None)
                source = (article.get('source') or '').strip()
                journal = (article.get('journalTitle') or '').strip()
                is_oa = (article.get('isOpenAccess', 'N') == 'Y')
                pub_year, pub_date = parse_date(article)

                if not title:
                    # skip entries without title
                    continue

                # Added lines ~223
                if pmcid in seen_pmids:
                    # Already have this article
                    continue
                seen_pmids.add(pmcid)

                article_data = {
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'pmcid': pmcid,
                    'source': source,
                    'journal_title': journal,
                    'is_open_access': is_oa,
                    'published_year': pub_year,
                    'published_date': pub_date.isoformat() if pub_date else None
                }

                all_articles.append(article_data)
                collected += 1

            if collected >= max_results:
                logger.info(f"Reached max_results={max_results} for {query}, stopping search.")
                break

            # Added lines ~246
            # If no new articles were added in this iteration, break out
            if len(all_articles) == before_add_count:
                break

            page += 1
            # Polite pause
            time.sleep(1)

        self.stdout.write(self.style.NOTICE(f"Done fetching {collected} articles for {microbe.binomial_name}"))
        logger.info(f"Done fetching {collected} articles for {microbe.binomial_name}")
        return all_articles
