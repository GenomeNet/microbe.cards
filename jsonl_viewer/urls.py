from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('entry/<str:species>/', views.entry_detail, name='entry_detail'),
]