# Generated by Django 5.1.1 on 2024-09-10 12:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jsonl_viewer', '0005_alter_modelranking_balanced_accuracy_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='phenotype',
            name='value',
            field=models.TextField(),
        ),
    ]
