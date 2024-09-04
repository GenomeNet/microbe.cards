# Generated by Django 5.1.1 on 2024-09-04 21:31

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='JSONLEntry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('custom_id', models.CharField(max_length=255, unique=True)),
                ('name', models.CharField(max_length=255)),
                ('content', models.JSONField()),
            ],
        ),
    ]
