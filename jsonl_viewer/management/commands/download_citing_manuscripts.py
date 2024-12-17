import os
import time
import requests
import json
import datetime
import logging
from django.core.management.base import BaseCommand
from django.conf import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('download_citing_manuscripts.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PAGE_SIZE = 25
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

def parse_date(article):
    pub_date_str = article.get('firstPublicationDate', '')
    pub_date = None
    if pub_date_str:
        try:
            pub_date = datetime.date.fromisoformat(pub_date_str)
        except ValueError:
            pass
    return pub_date

def fetch_citing_manuscripts(pmcid, max_results=100):
    query = f"cites:{pmcid}"
    page = 1
    collected = 0
    all_articles = []
    seen_pmids = set()

    while True:
        params = {
            'query': query,
            'format': 'json',
            'page': page,
            'pageSize': PAGE_SIZE,
            'resultType': 'core'  
        }
        try:
            response = robust_request(f"{BASE_URL}/search", params=params)
        except RuntimeError as e:
            logger.error(f"Error querying citing manuscripts for {pmcid}: {e}")
            break

        data = response.json()
        results = data.get('resultList', {}).get('result', [])
        if not results:
            break

        before_add_count = len(all_articles)
        for article in results:
            if collected >= max_results:
                break

            title = (article.get('title') or '').strip()
            abstract = (article.get('abstractText') or '').strip()
            authors_data = article.get('authorList', {}).get('author', [])
            authors = [a.get('fullName', '').strip() for a in authors_data if a.get('fullName')]
            citing_pmcid = article.get('pmcid', None)
            pmid = article.get('pmid', None)
            journal = (article.get('journalTitle') or '').strip()
            pub_date = parse_date(article)

            # Skip entries without title
            if not title:
                continue

            # Avoid duplicates by pmid or pmcid
            unique_id = pmid if pmid else citing_pmcid
            if unique_id in seen_pmids:
                continue
            seen_pmids.add(unique_id)

            article_data = {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'pmcid': citing_pmcid,
                'pmid': pmid,
                'journal_title': journal,
                'published_date': pub_date.isoformat() if pub_date else None
            }

            all_articles.append(article_data)
            collected += 1

        if collected >= max_results:
            logger.info(f"Reached max_results={max_results} for {pmcid}, stopping search.")
            break

        # If no new articles were added in this iteration, break
        if len(all_articles) == before_add_count:
            break

        page += 1
        # Polite pause
        time.sleep(1)

    logger.info(f"Fetched {collected} citing articles for {pmcid}")
    return all_articles

class Command(BaseCommand):
    help = (
        "Download citing manuscripts from Europe PMC for given PMCIDs "
        "and save them as JSON files."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--pmcid',
            help='A single PMCID to fetch citing manuscripts for.'
        )
        parser.add_argument(
            '--pmcids-file',
            help='A file containing one PMCID per line to fetch citing manuscripts for.'
        )
        parser.add_argument(
            '--output-dir',
            default='data/citing_manuscripts_raw',
            help='Directory to store the downloaded JSON files.'
        )
        parser.add_argument(
            '--max-results',
            type=int,
            default=100,
            help='Maximum number of citing manuscripts to download per PMCID.'
        )

    def handle(self, *args, **options):
        output_dir = options['output_dir']
        max_results = options['max_results']
        os.makedirs(output_dir, exist_ok=True)

        pmcids = []
        if options['pmcid']:
            pmcids.append(options['pmcid'])
        if options['pmcids_file']:
            with open(options['pmcids_file'], 'r', encoding='utf-8') as f:
                pmcids.extend(line.strip() for line in f if line.strip())

        if not pmcids:
            self.stdout.write(self.style.ERROR("No PMCID provided. Use --pmcid or --pmcids-file."))
            return

        start_time = time.time()
        self.stdout.write(self.style.NOTICE(f"Starting citing manuscripts download for {len(pmcids)} PMCID(s)..."))
        logger.info(f"Starting citing manuscripts download for {len(pmcids)} PMCID(s)...")

        total_collected = 0
        processed_pmcids = 0

        for pmcid in pmcids:
            json_path = os.path.join(output_dir, f"{pmcid}.json")

            # Skip if already exists
            if os.path.exists(json_path):
                self.stdout.write(self.style.WARNING(f"{pmcid} JSON already exists, skipping download."))
                logger.info(f"{pmcid} JSON already exists, skipping download.")
                processed_pmcids += 1
                continue

            self.stdout.write(self.style.NOTICE(f"Downloading citing manuscripts for {pmcid}"))
            logger.info(f"Downloading citing manuscripts for {pmcid}")

            articles = fetch_citing_manuscripts(pmcid, max_results=max_results)
            total_collected += len(articles)

            # Save to JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2)
            self.stdout.write(self.style.SUCCESS(f"Saved {len(articles)} citing manuscripts for {pmcid} to {json_path}"))
            logger.info(f"Saved {len(articles)} citing manuscripts for {pmcid} to {json_path}")

            processed_pmcids += 1

        elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f"Completed downloading citing manuscripts for {processed_pmcids} PMCID(s). "
            f"Total articles: {total_collected}. Elapsed: {elapsed:.2f}s"
        ))
        logger.info(f"Completed downloading citing manuscripts for {processed_pmcids} PMCID(s). "
                    f"Total articles: {total_collected}. Elapsed: {elapsed:.2f}s")