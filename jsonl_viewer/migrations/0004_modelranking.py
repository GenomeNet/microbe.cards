# Generated by Django 5.1.1 on 2024-09-10 10:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jsonl_viewer', '0003_alter_prediction_unique_together'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelRanking',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model', models.CharField(max_length=255)),
                ('target', models.CharField(max_length=255)),
                ('balanced_accuracy', models.FloatField()),
                ('precision', models.FloatField()),
                ('sample_size', models.IntegerField()),
            ],
            options={
                'unique_together': {('model', 'target')},
            },
        ),
    ]
