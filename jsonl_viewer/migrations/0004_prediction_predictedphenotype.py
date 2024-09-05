# Generated by Django 5.1.1 on 2024-09-05 21:57

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jsonl_viewer', '0003_microbe_phenotype_phenotypedefinition_taxonomy_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('prediction_id', models.CharField(max_length=100, unique=True)),
                ('model', models.CharField(max_length=100)),
                ('inference_date_time', models.DateTimeField()),
                ('microbe', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='predictions', to='jsonl_viewer.microbe')),
            ],
        ),
        migrations.CreateModel(
            name='PredictedPhenotype',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.JSONField()),
                ('definition', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='jsonl_viewer.phenotypedefinition')),
                ('prediction', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='predicted_phenotypes', to='jsonl_viewer.prediction')),
            ],
        ),
    ]
