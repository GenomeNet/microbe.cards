python manage.py makemigrations jsonl_viewer


python manage.py load_jsonl data/batch_6SH8sKb44rRryQhBim3q1MLX_output.jsonl

python manage.py load_ground_truth data/TableS1.csv

python manage.py runserver