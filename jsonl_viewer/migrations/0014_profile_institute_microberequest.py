# Generated by Django 5.1.1 on 2024-10-28 12:04

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jsonl_viewer', '0013_microbe_access_count_tools_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='institute',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.CreateModel(
            name='MicrobeRequest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('binomial_name', models.CharField(max_length=255, unique=True)),
                ('institute', models.CharField(blank=True, max_length=255, null=True)),
                ('email', models.EmailField(max_length=254)),
                ('status', models.CharField(choices=[('under_review', 'Under Review'), ('processing', 'Processing'), ('published', 'Published')], default='under_review', max_length=20)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('comments', models.TextField(blank=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='microbe_requests', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
