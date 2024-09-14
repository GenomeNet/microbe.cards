from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('microbe/<int:microbe_id>/', views.microbe_detail, name='microbe_detail'),
    path('model-ranking/', views.model_ranking, name='model_ranking'),
    path('about/', views.about, name='about'),
    path('imprint/', views.imprint, name='imprint'),
    path('taxonomy-autocomplete/', views.taxonomy_autocomplete, name='taxonomy_autocomplete'),
    path('phenotype-autocomplete/', views.phenotype_autocomplete, name='phenotype_autocomplete'),
    path('model/<path:model_name>/', views.model_detail, name='model_detail'),
]