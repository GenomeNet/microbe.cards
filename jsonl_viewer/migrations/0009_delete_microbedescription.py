# Generated by Django 5.1.1 on 2024-09-17 13:43

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('jsonl_viewer', '0008_alter_microbedescription_microbe'),
    ]

    operations = [
        migrations.DeleteModel(
            name='MicrobeDescription',
        ),
    ]