# Generated by Django 5.1.1 on 2024-09-17 11:47

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jsonl_viewer', '0007_alter_phenotype_value_microbedescription'),
    ]

    operations = [
        migrations.AlterField(
            model_name='microbedescription',
            name='microbe',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='descriptions', to='jsonl_viewer.microbe'),
        ),
    ]
