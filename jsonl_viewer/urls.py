from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('microbe/<int:microbe_id>/', views.microbe_detail, name='microbe_detail'),
    path('model-ranking/', views.model_ranking, name='model_ranking'),
    path('about/', views.about, name='about'),
    path('imprint/', views.imprint, name='imprint'),
    path('search/', views.search, name='search'),
    path('taxonomy-autocomplete/', views.taxonomy_autocomplete, name='taxonomy_autocomplete'),
]

