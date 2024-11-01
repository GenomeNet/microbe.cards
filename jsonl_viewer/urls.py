from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('microbe/<int:microbe_id>/', views.microbe_detail, name='microbe_detail'),
    path('model/<path:model_name>/', views.model_detail, name='model_detail'),
    path('model-ranking/', views.model_ranking, name='model_ranking'),
    path('about/', views.about, name='about'),
    path('about/technical/', views.about_llm, name='about_llm'),
    path('imprint/', views.imprint, name='imprint'),
    path('dataprotection/', views.dataprotection, name='dataprotection'),
    path('download/', views.download, name='download'),
    path('search/', views.search_microbes, name='search_microbes'),
    path('browse/', views.browse_microbes, name='browse_microbes'),
    path('request/', views.request_microbes, name='request_microbe'),
    path('reviewer-login/<str:token>/', views.reviewer_login, name='demo-user'),
    path('report-error/', views.report_error, name='report_error'),
    path('profile/settings/', views.profile_settings, name='profile_settings'),
    path('download/database/', views.download_database, name='download_database'),  
    
    # Authentication URLs
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
     path('home/', views.home, name='home'), 

    
    # Star/Unstar URL
    path('toggle-star/<int:microbe_id>/', views.toggle_star, name='toggle_star'),

    # Autocomplete URLs
    path('taxonomy-autocomplete/', views.taxonomy_autocomplete, name='taxonomy_autocomplete'),
    path('phenotype-autocomplete/', views.phenotype_autocomplete, name='phenotype_autocomplete'),
    
    # Profile Settings URL
    path('settings/', views.profile_settings, name='profile_settings'),
]
