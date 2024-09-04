from django.core.management.base import BaseCommand
from jsonl_viewer.models import JSONLEntry

class Command(BaseCommand):
    help = 'Load data from JSONL file into database'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to the JSONL file')

    def handle(self, *args, **options):
        file_path = options['file_path']
        JSONLEntry.load_from_jsonl(file_path)
        self.stdout.write(self.style.SUCCESS('Successfully loaded JSONL data'))