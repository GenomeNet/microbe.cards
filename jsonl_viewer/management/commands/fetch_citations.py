import time
import requests
import logging
from django.core.management.base import BaseCommand
from django.utils import timezone
from jsonl_viewer.models import Manuscript

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('fetch_citations.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PAGE_SIZE = 1
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

def fetch_citations_for_pmcid(pmcid):
    query = f"pmcid:{pmcid}"
    params = {
        'query': query,
        'format': 'json',
        'page': 1,
        'pageSize': PAGE_SIZE,
        'resultType': 'core'
    }
    response = robust_request(f"{BASE_URL}/search", params=params)
    data = response.json()
    results = data.get('resultList', {}).get('result', [])
    if not results:
        return None
    article_data = results[0]
    return article_data.get('citedByCount', None)

class Command(BaseCommand):
    help = "Fetch citation counts from Europe PMC for manuscripts with PMCID and store them in the database."

    def add_arguments(self, parser):
        parser.add_argument(
            '--max',
            type=int,
            default=1000,
            help='Maximum number of manuscripts to process in one run.'
        )

    def handle(self, *args, **options):
        max_to_process = options['max']
        
        manuscripts = Manuscript.objects.exclude(pmcid__isnull=True).exclude(pmcid__exact='')[:max_to_process]

        self.stdout.write(self.style.NOTICE(f"Found {manuscripts.count()} manuscripts with PMCID to process."))
        logger.info(f"Found {manuscripts.count()} manuscripts with PMCID to process.")

        updated_count = 0

        for manuscript in manuscripts:
            pmcid = manuscript.pmcid
            self.stdout.write(self.style.NOTICE(f"Fetching citations for PMCID: {pmcid}"))
            logger.info(f"Fetching citations for PMCID: {pmcid}")

            try:
                cited_by_count = fetch_citations_for_pmcid(pmcid)
            except RuntimeError as e:
                self.stdout.write(self.style.ERROR(f"Error fetching citation for {pmcid}: {e}"))
                logger.error(f"Error fetching citation for {pmcid}: {e}")
                continue

            if cited_by_count is not None:
                if manuscript.citation_count != cited_by_count:
                    manuscript.citation_count = cited_by_count
                    manuscript.citation_updated_at = timezone.now()
                    manuscript.save()
                    updated_count += 1
                    self.stdout.write(self.style.SUCCESS(f"Updated citation count for {pmcid} to {cited_by_count}"))
                    logger.info(f"Updated citation count for {pmcid} to {cited_by_count}")
                else:
                    self.stdout.write(self.style.WARNING(f"No change in citation count for {pmcid}. Current: {manuscript.citation_count}"))
            else:
                self.stdout.write(self.style.WARNING(f"No citation data found for {pmcid}"))
                logger.warning(f"No citation data found for {pmcid}")

        self.stdout.write(self.style.SUCCESS(
            f"Completed processing. Updated citation counts for {updated_count} manuscripts."
        ))
        logger.info(f"Completed processing. Updated citation counts for {updated_count} manuscripts.")