from django.contrib import admin
from django.urls import path, include
from jsonl_viewer import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('jsonl_viewer.urls')),
    path('model_ranking/', views.model_ranking, name='model_ranking'),
]