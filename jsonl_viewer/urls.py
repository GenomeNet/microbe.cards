from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('microbe/<int:microbe_id>/', views.microbe_detail, name='microbe_detail'),
    path('model-ranking/', views.model_ranking, name='model_ranking'),
    path('about/', views.about, name='about'),  # Ensure this exists or remove based on your needs
    path('imprint/', views.imprint, name='imprint'),
    path('taxonomy-autocomplete/', views.taxonomy_autocomplete, name='taxonomy_autocomplete'),
    path('phenotype-autocomplete/', views.phenotype_autocomplete, name='phenotype_autocomplete'),
    path('model/<path:model_name>/', views.model_detail, name='model_detail'),
    path('dataprotection/', views.dataprotection, name='dataprotection'),
    path('browse/', views.browse_microbes, name='browse_microbes'),  # Ensure this exists
    path('report-error/', views.report_error, name='report_error'),  # Add this line
]