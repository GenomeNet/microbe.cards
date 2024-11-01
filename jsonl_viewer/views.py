import os
import json
import random
import hashlib
import time
import datetime
from django.db import connection
from django.conf import settings
from datetime import timedelta 
from django.utils import timezone
from django.db.models import Count, Prefetch, Q, Exists, OuterRef, BooleanField, Value, TextField, Subquery, OuterRef
from functools import reduce
from operator import or_
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.core.paginator import Paginator
from django.db import models, transaction
from collections import defaultdict, OrderedDict, Counter
from .models import (
    Microbe, Phenotype, PhenotypeDefinition, Prediction, PredictedPhenotype,
    ModelRanking, Taxonomy, MicrobeDescription, ErrorReport, MicrobeAccess,
    MicrobeRequest, PhenotypeSummary
)
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.urls import reverse
import logging
from django.template.defaultfilters import register
from urllib.parse import unquote
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.shortcuts import redirect
from .forms import CustomUserCreationForm
from django.contrib.auth.models import User
import csv
import uuid
from django.core.cache import cache


logger = logging.getLogger(__name__)

# Store tokens securely; this is a simplistic example
REVIEWER_TOKENS = {
    'not-a-secure-password': 'demo-user',
}


def get_cache_key(request_get):
    """Generate a unique cache key based on search parameters"""
    # Sort the items to ensure consistent ordering
    sorted_params = sorted(request_get.items())
    # Create a string representation of the parameters
    param_string = ','.join(f'{k}:{v}' for k, v in sorted_params)
    # Create a hash of the parameters
    return f'search_results_{hashlib.md5(param_string.encode()).hexdigest()}'


def calculate_similarity(phenotypes1, phenotypes2):
    """Calculate similarity score between two sets of phenotypes"""
    common_phenotypes = set(phenotypes1.keys()) & set(phenotypes2.keys())
    if not common_phenotypes:
        return 0
        
    matches = sum(1 for p in common_phenotypes 
                 if phenotypes1[p] == phenotypes2[p] 
                 and phenotypes1[p] not in [None, '', 'N/A']
                 and phenotypes2[p] not in [None, '', 'N/A'])
    
    return matches / len(common_phenotypes) if common_phenotypes else 0

def get_majority_phenotypes_batch(microbe, phenotype_definitions):
    """
    Get majority phenotypes from PhenotypeSummary for a microbe
    """
    phenotype_summaries = PhenotypeSummary.objects.filter(
        microbe=microbe,
        definition__in=phenotype_definitions
    ).select_related('definition')
    
    # Create a dictionary of phenotype summaries
    majority_phenotypes = {}
    for summary in phenotype_summaries:
        if summary.majority_value not in [None, '', 'N/A']:
            majority_phenotypes[summary.definition.name] = {
                'value': summary.majority_value,
                'agreement': summary.agreement_percentage,
                'supporting_models': summary.supporting_models,
                'total_models': summary.total_models
            }
    
    return majority_phenotypes


def get_similar_microbes(microbe_id, phenotype_definitions):
    """
    Get similar microbes with caching
    """
    cache_key = f'similar_microbes_{microbe_id}'
    cached_results = cache.get(cache_key)
    
    if cached_results is not None:
        return cached_results
        
    current_microbe = Microbe.objects.get(id=microbe_id)
    
    # Get current microbe's phenotype summaries
    current_summaries = PhenotypeSummary.objects.filter(
        microbe=current_microbe,
        definition__in=phenotype_definitions
    ).select_related('definition')
    
    # Create a dictionary of current microbe's majority values
    current_values = {
        summary.definition.name: summary.majority_value 
        for summary in current_summaries 
        if summary.majority_value not in [None, '', 'N/A']
    }
    
    # Use raw SQL for better performance
    similar_microbes = []
    with connection.cursor() as cursor:
        placeholders = ','.join(['%s'] * len(current_values))
        phenotype_names = list(current_values.keys())
        
        query = f"""
            WITH matching_counts AS (
                SELECT 
                    m.id,
                    m.binomial_name,
                    COUNT(*) as match_count,
                    SUM(CASE WHEN ps.majority_value IS NOT NULL 
                        AND ps.majority_value != '' 
                        AND ps.majority_value != 'N/A' THEN 1 ELSE 0 END) as total_comparable
                FROM jsonl_viewer_microbe m
                JOIN jsonl_viewer_phenotypesummary ps ON m.id = ps.microbe_id
                JOIN jsonl_viewer_phenotypedefinition pd ON ps.definition_id = pd.id
                WHERE pd.name IN ({placeholders})
                AND m.id != %s
                GROUP BY m.id, m.binomial_name
                HAVING total_comparable > 0
            )
            SELECT 
                id,
                binomial_name,
                match_count,
                total_comparable,
                CAST(match_count AS FLOAT) / total_comparable as similarity
            FROM matching_counts
            WHERE CAST(match_count AS FLOAT) / total_comparable >= 0.5
            ORDER BY similarity DESC, match_count DESC
            LIMIT 5
        """
        
        cursor.execute(query, [*phenotype_names, microbe_id])
        results = cursor.fetchall()
        
        for row in results:
            similar_microbes.append({
                'microbe': Microbe.objects.get(id=row[0]),
                'similarity': row[4],
                'matching_phenotypes': row[2],
                'total_comparable': row[3]
            })
    
    # Cache the results for 24 hours
    cache.set(cache_key, similar_microbes, 60 * 60 * 24)
    return similar_microbes

def reviewer_login(request, token):
       username = REVIEWER_TOKENS.get(token)
       if username:
           try:
               user = User.objects.get(username=username)
               login(request, user)
               #messages.success(request, "Reviewer logged in successfully.")
               return redirect('home')
           except User.DoesNotExist:
               messages.error(request, "User does not exist.")
               return redirect('login')
       else:
           # Handle invalid token
           messages.error(request, "Invalid login token.")
           return redirect('login')
         
@register.filter
def subtract(value, arg):
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

@login_required
def profile_settings(request):
    if request.method == 'POST':
        # Check which form is being submitted
        if 'binomial_name' in request.POST:
            # Handle Microbe Request Form
            binomial_name = request.POST.get('binomial_name').strip()
            if binomial_name:
                # Check if the request already exists
                existing_request = MicrobeRequest.objects.filter(
                    binomial_name__iexact=binomial_name,
                    user=request.user,
                    status__in=['under_review', 'processing']
                ).exists()
                if not existing_request:
                    MicrobeRequest.objects.create(
                        binomial_name=binomial_name,
                        user=request.user,
                        institute=request.user.profile.institute,
                        email=request.user.email,
                        status='under_review'
                    )
                    messages.success(request, f"Microbe '{binomial_name}' has been requested successfully.")
                else:
                    messages.warning(request, f"You already have a pending request for '{binomial_name}'.")
            else:
                messages.error(request, "Please enter a valid Binomial Name.")
            return redirect('profile_settings')
        else:
            # Handle Profile Update Form
            email = request.POST.get('email').strip()
            institute = request.POST.get('institute').strip()
            if email:
                request.user.email = email
                request.user.save()
                request.user.profile.institute = institute
                request.user.profile.save()
                messages.success(request, "Your profile has been updated successfully.")
            else:
                messages.error(request, "Email is required.")
            return redirect('profile_settings')
    else:
        microbe_requests = request.user.microbe_requests.all().order_by('-created_at')
        return render(request, 'jsonl_viewer/profile_settings.html', {
            'microbe_requests': microbe_requests
        })

@login_required
def toggle_star(request, microbe_id):
    microbe = get_object_or_404(Microbe, id=microbe_id)
    profile = request.user.profile
    if microbe in profile.starred_microbes.all():
        profile.starred_microbes.remove(microbe)
       
    else:
        profile.starred_microbes.add(microbe)
        
    return redirect('microbe_detail', microbe_id=microbe_id)

def index(request):
    # Get parameters from the request first
    include_no_predictions = request.POST.get('include_no_predictions', 'false') == 'true'
    selected_phenotype = request.POST.get('selected_phenotype', '')
    selected_value = request.POST.get('selected_value', '')
    search_term = request.POST.get('search_term', '').strip()

    # Aggregate statistics
    total_species_with_ground_truth = Microbe.objects.filter(
        phenotype__value__isnull=False
    ).distinct().count()
    
    total_species_with_predictions = Microbe.objects.filter(
        phenotype_summaries__majority_value__isnull=False
    ).distinct().count()
    
    total_phenotypes = PhenotypeDefinition.objects.count()
    total_models = Prediction.objects.values('model').distinct().count()

    # Initialize the microbes queryset
    microbes = Microbe.objects.select_related('taxonomy')
    search_results = None

    # Apply filters
    if not include_no_predictions:
        microbes = microbes.filter(phenotype_summaries__isnull=False).distinct()

    if search_term:
        microbes = microbes.filter(
            Q(binomial_name__icontains=search_term) |
            Q(taxonomy__superkingdom__icontains=search_term) |
            Q(taxonomy__phylum__icontains=search_term) |
            Q(taxonomy__class_name__icontains=search_term) |
            Q(taxonomy__order__icontains=search_term) |
            Q(taxonomy__family__icontains=search_term) |
            Q(taxonomy__genus__icontains=search_term) |
            Q(taxonomy__species__icontains=search_term)
        ).distinct()

    if selected_phenotype:
        phenotype_def = PhenotypeDefinition.objects.filter(name=selected_phenotype).first()
        if phenotype_def:
            if selected_value:
                microbes = microbes.filter(
                    phenotype_summaries__definition=phenotype_def,
                    phenotype_summaries__majority_value__iexact=selected_value
                ).distinct()
            else:
                microbes = microbes.filter(
                    phenotype_summaries__definition=phenotype_def
                ).distinct()

    phenotype_definitions = PhenotypeDefinition.objects.all()

    if request.method == 'POST':
        # Prefetch related data to optimize queries
        microbes = microbes.prefetch_related(
            Prefetch('phenotype_set', queryset=Phenotype.objects.exclude(value__in=[None, '', 'N/A']).select_related('definition')),
            Prefetch('phenotype_summaries', queryset=PhenotypeSummary.objects.select_related('definition')),
            Prefetch('descriptions')
        )

        phenotype_definitions_all = list(phenotype_definitions)

        for microbe in microbes:
            # Ground truth phenotypes
            gt_phenotypes = {p.definition.id: p for p in microbe.phenotype_set.all()}
            microbe.ground_truth_count = len(gt_phenotypes)

            # Get prediction summaries
            prediction_summaries = {ps.definition.id: ps for ps in microbe.phenotype_summaries.all()}
            microbe.additional_predictions_count = len(set(prediction_summaries.keys()) - set(gt_phenotypes.keys()))

            # Phenotypes availability
            phenotypes_available = []
            for pd in phenotype_definitions_all:
                if pd.name == "Member of WA subset":
                    continue
                has_gt = pd.id in gt_phenotypes
                has_prediction = pd.id in prediction_summaries and prediction_summaries[pd.id].majority_value not in [None, '', 'N/A']
                phenotypes_available.append({
                    'definition': pd,
                    'has_gt': has_gt,
                    'has_prediction': has_prediction,
                })
            microbe.phenotypes_available = phenotypes_available

        search_results = microbes

    # ----- Microbe of the Day Logic -----
    today = datetime.date.today()
    random_seed = today.toordinal()
    random_instance = random.Random(random_seed)

    eligible_microbes = Microbe.objects.annotate(
        description_count=Count('descriptions', filter=Q(descriptions__description_type="General Information")),
        prediction_summary_count=Count('phenotype_summaries')
    ).filter(
        description_count__gte=1,
        prediction_summary_count__gte=1
    )

    eligible_microbes_list = list(eligible_microbes)
    microbe_of_the_day = random_instance.choice(eligible_microbes_list) if eligible_microbes_list else None

    if microbe_of_the_day:
        general_description = microbe_of_the_day.descriptions.filter(description_type="General Information").first()
        description = general_description.description if general_description else "No description available."

        gt_phenotypes = {p.definition.id: p for p in microbe_of_the_day.phenotype_set.all()}
        prediction_summaries = {ps.definition.id: ps for ps in microbe_of_the_day.phenotype_summaries.all()}

        phenotypes_available = []
        phenotype_definitions_all = list(PhenotypeDefinition.objects.all())
        for pd in phenotype_definitions_all:
            if pd.name == "Member of WA subset":
                continue
            has_gt = pd.id in gt_phenotypes
            has_prediction = pd.id in prediction_summaries and prediction_summaries[pd.id].majority_value not in [None, '', 'N/A']
            phenotypes_available.append({
                'definition': pd,
                'has_gt': has_gt,
                'has_prediction': has_prediction,
            })

        microbe_of_the_day.description = description
        microbe_of_the_day.phenotypes_available = phenotypes_available

    # ----- Trending Microbes Calculation -----
    today = timezone.now().date()
    thirty_days_ago = today - timedelta(days=29)
    fifteen_days_ago = today - timedelta(days=14)

    # Rest of your trending microbes code remains the same
    recent_counts = MicrobeAccess.objects.filter(
        accessed_at__date__gte=fifteen_days_ago
    ).values('microbe__id', 'microbe__binomial_name').annotate(
        count=Count('id')
    )

    previous_counts = MicrobeAccess.objects.filter(
        accessed_at__date__gte=thirty_days_ago,
        accessed_at__date__lt=fifteen_days_ago
    ).values('microbe__id', 'microbe__binomial_name').annotate(
        count=Count('id')
    )

    previous_dict = {item['microbe__id']: item['count'] for item in previous_counts}

    trending_data = []
    for recent in recent_counts:
        microbe_id = recent['microbe__id']
        recent_count = recent['count']
        previous_count = max(previous_dict.get(microbe_id, 0), 1)
        
        increase = ((recent_count - previous_count) / previous_count) * 100
        trending_data.append({
            'name': recent['microbe__binomial_name'],
            'increase': round(increase),
            'id': microbe_id
        })

    trending_data = sorted(trending_data, key=lambda x: abs(x['increase']), reverse=True)[:6]
    
    max_increase = max(abs(item['increase']) for item in trending_data) if trending_data else 1
    for item in trending_data:
        relative = abs(item['increase']) / max_increase
        item['intensity'] = round(0.1 + (relative * 0.2), 2)

        access_counts = MicrobeAccess.objects.filter(
            microbe__id=item['id'],
            accessed_at__date__gte=thirty_days_ago
        ).extra({'date': "date(accessed_at)"}).values('date').annotate(count=Count('id')).order_by('date')

        date_list = [thirty_days_ago + timedelta(days=x) for x in range(30)]
        daily_dates = [date.strftime('%Y-%m-%d') for date in date_list]
        access_dict = {access['date']: access['count'] for access in access_counts}
        daily_access_counts = [access_dict.get(date, 0) for date in date_list]

        item['daily_dates'] = daily_dates
        item['daily_access_counts'] = daily_access_counts

    context = {
        'total_species_with_ground_truth': total_species_with_ground_truth,
        'total_species_with_predictions': total_species_with_predictions,
        'total_phenotypes': total_phenotypes,
        'total_models': total_models,
        'search_results': search_results,
        'phenotype_definitions': phenotype_definitions,
        'include_no_predictions': include_no_predictions,
        'selected_phenotype': selected_phenotype,
        'selected_value': selected_value,
        'search_term': search_term,
        'microbe_of_the_day': microbe_of_the_day,
        'trending_data': trending_data,
    }

    return render(request, 'jsonl_viewer/index.html', context)

def taxonomy_autocomplete(request):
    query = request.GET.get('q', '')
    results = []
    if query:
        taxonomies = Taxonomy.objects.filter(
            Q(superkingdom__icontains=query) |
            Q(phylum__icontains=query) |
            Q(class_name__icontains=query) |
            Q(order__icontains=query) |
            Q(family__icontains=query) |
            Q(genus__icontains=query) |
            Q(species__icontains=query)
        ).distinct()

        suggestions = []
        for tax in taxonomies:
            if query.lower() in (tax.superkingdom or '').lower():
                suggestions.append({'text': tax.superkingdom, 'level': 'Superkingdom'})
            if query.lower() in (tax.phylum or '').lower():
                suggestions.append({'text': tax.phylum, 'level': 'Phylum'})
            if query.lower() in (tax.class_name or '').lower():
                suggestions.append({'text': tax.class_name, 'level': 'Class'})
            if query.lower() in (tax.order or '').lower():
                suggestions.append({'text': tax.order, 'level': 'Order'})
            if query.lower() in (tax.family or '').lower():
                suggestions.append({'text': tax.family, 'level': 'Family'})
            if query.lower() in (tax.genus or '').lower():
                suggestions.append({'text': tax.genus, 'level': 'Genus'})
            if query.lower() in (tax.species or '').lower():
                suggestions.append({'text': tax.species, 'level': 'Species'})

        microbes = Microbe.objects.filter(
            binomial_name__icontains=query
        ).values_list('binomial_name', flat=True).distinct()

        for microbe_name in microbes:
            suggestions.append({'text': microbe_name, 'level': 'Binomial Name'})

        unique_suggestions = { (s['text'], s['level']): s for s in suggestions }.values()

        results = [{'id': item['text'], 'text': item['text'], 'level': item['level']} for item in list(unique_suggestions)[:20]]

    return JsonResponse({'results': results})

def phenotype_autocomplete(request):
    query = request.GET.get('q', '')
    results = []
    if query:
        phenotypes = PhenotypeDefinition.objects.filter(
            name__icontains=query
        ).values('id', 'name')[:10]
        results = [{'id': p['id'], 'text': p['name']} for p in phenotypes]

    return JsonResponse({'results': results})

def microbe_detail(request, microbe_id):
    start_time = time.time()
    microbe = get_object_or_404(Microbe.objects.select_related('taxonomy'), id=microbe_id)
    
    # Get phenotype definitions and ground truth
    phenotype_definitions = PhenotypeDefinition.objects.all()
    microbe_phenotypes = Phenotype.objects.filter(microbe=microbe).select_related('definition')
    microbe_phenotypes_dict = {p.definition.id: p for p in microbe_phenotypes}
    
    # Get prediction summaries and individual predictions
    phenotype_summaries = PhenotypeSummary.objects.filter(
        microbe=microbe
    ).select_related('definition')
    
    # Get all predictions for this microbe
    predictions = Prediction.objects.filter(
        microbe=microbe
    ).prefetch_related('predicted_phenotypes')
    
    # Get all unique models
    models_used = list(predictions.values_list('model', flat=True).distinct())
    na_counts = {model: 0 for model in models_used}
    
    phenotype_data = []
    ground_truth_count = 0
    additional_predictions_count = 0

    for phenotype_def in phenotype_definitions:
        if phenotype_def.name == "Member of WA subset":
            continue
            
        # Get ground truth
        phenotype = microbe_phenotypes_dict.get(phenotype_def.id)
        ground_truth_value = phenotype.value if phenotype else None
        if ground_truth_value not in [None, '', 'N/A']:
            ground_truth_count += 1
            
        # Get prediction summary
        summary = next((s for s in phenotype_summaries if s.definition_id == phenotype_def.id), None)
        
        # Get individual model predictions for this phenotype
        model_predictions = []
        for prediction in predictions:
            predicted_phenotype = prediction.predicted_phenotypes.filter(definition=phenotype_def).first()
            predicted_value = predicted_phenotype.value if predicted_phenotype else None
            
            if predicted_value in [None, '', 'N/A']:
                na_counts[prediction.model] += 1
                
            model_performance = ModelRanking.objects.filter(
                model=prediction.model,
                target=phenotype_def.name
            ).first()
            
            model_predictions.append({
                'model': prediction.model,
                'predicted_value': predicted_value,
                'model_performance': model_performance,
                'inference_date': prediction.inference_date_time,
            })
        
        data = {
            'phenotype_definition': phenotype_def,
            'ground_truth': ground_truth_value,
            'prediction_summary': {
                'majority_value': summary.majority_value if summary else None,
                'agreement_percentage': summary.agreement_percentage if summary else 0,
                'supporting_models': summary.supporting_models if summary else 0,
                'total_models': summary.total_models if summary else 0
            },
            'predictions': model_predictions
        }
        
        phenotype_data.append(data)

    # Sort the models based on fewest 'N/A' counts
    sorted_models = sorted(models_used, key=lambda m: na_counts.get(m, 0))

    # Sort predictions within each phenotype data
    phenotype_data_sorted = []
    for data in phenotype_data:
        sorted_pred = sorted(
            data['predictions'],
            key=lambda p: sorted_models.index(p['model']) if p['model'] in sorted_models else len(sorted_models)
        )
        data['predictions'] = sorted_pred
        phenotype_data_sorted.append(data)

    # Count non-'N/A' predictions per model
    model_prediction_counts = {model: 0 for model in models_used}
    for data in phenotype_data_sorted:
        for pred in data['predictions']:
            if pred['predicted_value'] not in [None, '', 'N/A']:
                model_prediction_counts[pred['model']] += 1

    # Count additional predictions
    additional_predictions_count = 0
    for data in phenotype_data_sorted:
        if data['ground_truth'] in [None, '', 'N/A']:
            has_prediction = any(pred['predicted_value'] not in [None, '', 'N/A'] for pred in data['predictions'])
            if has_prediction:
                additional_predictions_count += 1

    # Retrieve MicrobeDescriptions
    descriptions = microbe.descriptions.all().order_by('description_type', 'model')
    descriptions_by_type = {}
    for desc in descriptions:
        if desc.description_type not in descriptions_by_type:
            descriptions_by_type[desc.description_type] = []
        descriptions_by_type[desc.description_type].append(desc)

    # Access Tracking
    if request.user.is_authenticated:
        MicrobeAccess.objects.create(microbe=microbe, user=request.user)
        microbe.access_count_users = models.F('access_count_users') + 1
    else:
        user_agent = request.META.get('HTTP_USER_AGENT', 'Unknown')
        MicrobeAccess.objects.create(microbe=microbe, search_tool=user_agent)
        microbe.access_count_tools = models.F('access_count_tools') + 1
    microbe.save(update_fields=['access_count_users', 'access_count_tools'])

    # Star Count Logic
    star_count = microbe.starred_by.count()
    user_has_starred = False
    if request.user.is_authenticated:
        user_has_starred = microbe in request.user.profile.starred_microbes.all()

    # Access History
    today = timezone.now().date()
    thirty_days_ago = today - timedelta(days=29)
    
    daily_accesses = MicrobeAccess.objects.filter(
        microbe=microbe,
        accessed_at__date__gte=thirty_days_ago
    ).extra({'date': "date(accessed_at)"}).values('date').annotate(count=Count('id')).order_by('date')

    date_list = [thirty_days_ago + timedelta(days=x) for x in range(30)]
    daily_dates = [date.strftime('%Y-%m-%d') for date in date_list]
    access_dict = {access['date']: access['count'] for access in daily_accesses}
    
    # Generate synthetic data if no real data exists
    random.seed(microbe.id)
    daily_access_counts = []
    
    for date in date_list:
        real_count = access_dict.get(date, 0)
        if real_count == 0:
            synthetic_count = random.choices(
                [0, 1, 2, 3, 4, 5],
                weights=[0.1, 0.2, 0.3, 0.3, 0.05, 0.05]
            )[0]
            daily_access_counts.append(synthetic_count)
        else:
            daily_access_counts.append(real_count)

    # Add noise
    for i in range(len(daily_access_counts)):
        if random.random() < 0.1:
            daily_access_counts[i] += random.randint(1, 2)

    # Similar microbes
    logger.info("Starting similar microbes calculation")
    phenotype_definitions = list(PhenotypeDefinition.objects.exclude(name="Member of WA subset"))
    similar_microbes = get_similar_microbes(microbe_id, phenotype_definitions)

    context = {
        'microbe': microbe,
        'phenotype_data': phenotype_data_sorted,
        'ground_truth_count': ground_truth_count,
        'model_prediction_counts': model_prediction_counts,
        'additional_predictions_count': additional_predictions_count,
        'descriptions_by_type': descriptions_by_type,
        'watch_count': star_count,
        'user_has_starred': user_has_starred,
        'daily_dates': daily_dates,
        'daily_access_counts': daily_access_counts,
        'similar_microbes': similar_microbes,
        'execution_time': time.time() - start_time
    }
    
    return render(request, 'jsonl_viewer/card.html', context)

def model_detail(request, model_name):
    model_name = unquote(model_name)
    rankings = ModelRanking.objects.filter(model=model_name)

    if not rankings.exists():
        return render(request, 'jsonl_viewer/model_not_found.html', {'model_name': model_name})

    context = {
        'model_name': model_name,
        'rankings': rankings,
    }
    return render(request, 'jsonl_viewer/model_detail.html', context)


def model_ranking(request):
    rankings = {}
    phenotype_descriptions = {}
    for ranking in ModelRanking.objects.all():
        if ranking.target not in rankings:
            rankings[ranking.target] = {}
            phenotype_def = PhenotypeDefinition.objects.filter(name=ranking.target).first()
            phenotype_descriptions[ranking.target] = phenotype_def.description if phenotype_def else ''
        rankings[ranking.target][ranking.model] = {
            'balanced_accuracy': ranking.balanced_accuracy,
            'precision': ranking.precision,
            'sample_size': ranking.sample_size
        }

    sorted_rankings = {}
    for phenotype, models in rankings.items():
        sorted_rankings[phenotype] = OrderedDict(
            sorted(
                models.items(),
                key=lambda x: x[1]['balanced_accuracy'] if x[1]['balanced_accuracy'] is not None else -1,
                reverse=True
            )
        )

    logger.debug("Rankings: %s", sorted_rankings)
    logger.debug("Phenotype Descriptions: %s", phenotype_descriptions)

    context = {
        'rankings': sorted_rankings,
        'phenotype_descriptions': phenotype_descriptions,
    }

    return render(request, 'jsonl_viewer/model_ranking.html', context)


def search_microbes(request):
    results = Microbe.objects.select_related('taxonomy').all()
    active_filters = []
    
    # Text search
    text_search = request.GET.get('text_search', '').strip()
    if text_search:
        results = results.filter(descriptions__description__icontains=text_search)

    # Get phenotype definitions for the filter dropdowns
    phenotype_definitions = PhenotypeDefinition.objects.all()
    phenotype_definitions_json = json.dumps([{
        'id': pd.id,
        'name': pd.name,
        'description': pd.description,
        'data_type': pd.data_type,
        'allowed_values': json.loads(pd.allowed_values) if pd.allowed_values else []
    } for pd in phenotype_definitions])

    # Phenotype filtering
    index = 0
    phenotype_filters = defaultdict(list)  # Maps phenotype_id to list of values

    while f'phenotype_{index}' in request.GET:
        phenotype_id = request.GET.get(f'phenotype_{index}')
        filter_value = request.GET.get(f'value_{index}')
        
        if phenotype_id and filter_value:
            filter_value_stripped = filter_value.strip('"\'')
            phenotype_filters[phenotype_id].append(filter_value_stripped)
            
            # Add to active filters for display
            phenotype_def = PhenotypeDefinition.objects.get(id=phenotype_id)
            active_filters.append({
                'phenotype_id': phenotype_id,
                'phenotype_name': phenotype_def.name,
                'value': filter_value_stripped
            })
        index += 1

    # Build conditions
    filter_conditions = []
    for phenotype_id, values in phenotype_filters.items():
        conditions = Q()
        for value in values:
            if value.lower() in ['true', 'false']:
                conditions |= Q(
                    phenotype_summaries__definition_id=phenotype_id,
                    phenotype_summaries__majority_value=value.lower()
                )
            else:
                conditions |= Q(
                    phenotype_summaries__definition_id=phenotype_id,
                    phenotype_summaries__majority_value__icontains=value
                )
        filter_conditions.append(conditions)

    # Apply all filters
    if filter_conditions:
        final_filter = reduce(lambda x, y: x & y, filter_conditions)
        results = results.filter(final_filter)

    results = results.distinct()


    # Organize results hierarchically
    organized_results = {}
    for microbe in results:
        superkingdom = microbe.taxonomy.superkingdom or 'Unknown Superkingdom'
        phylum = microbe.taxonomy.phylum or 'Unknown Phylum'
        
        if superkingdom not in organized_results:
            organized_results[superkingdom] = {}
        if phylum not in organized_results[superkingdom]:
            organized_results[superkingdom][phylum] = []
        organized_results[superkingdom][phylum].append(microbe)

    # Sort results
    organized_results = dict(sorted(organized_results.items()))
    for superkingdom in organized_results:
        organized_results[superkingdom] = dict(sorted(organized_results[superkingdom].items()))

    context = {
        'organized_results': organized_results,
        'result_count': results.count(),
        'phenotype_definitions': phenotype_definitions,
        'phenotype_definitions_json': phenotype_definitions_json,
        'active_filters': active_filters,
    }
    
    return render(request, 'jsonl_viewer/search.html', context)

def about(request):
    return render(request, 'jsonl_viewer/about.html')

def download(request):
    return render(request, 'jsonl_viewer/download.html')

def about_llm(request):
    return render(request, 'jsonl_viewer/about_llm.html')

def imprint(request):
   return render(request, 'jsonl_viewer/imprint.html')
    
def dataprotection(request):
   return render(request, 'jsonl_viewer/dataprotection.html')
    
def request_microbes(request):
   return render(request, 'jsonl_viewer/request.html')

def search(request):
    logger.info("Search view called")
    results = None
    phenotype_definitions = PhenotypeDefinition.objects.all()
    phenotype_definitions_json = json.dumps([{
        'id': pd.id,
        'name': pd.name,
        'description': pd.description,
        'data_type': pd.data_type,
        'allowed_values': pd.allowed_values  # This is already a list since it's a JSONField
    } for pd in phenotype_definitions])
    if request.method == 'POST':
        logger.info("POST request received")
        taxonomy_level = request.POST.get('taxonomy_level', '')
        taxonomy_value = request.POST.get('taxonomy_value', '')
        phenotype_ids = request.POST.getlist('phenotype_ids[]')
        phenotype_values = request.POST.getlist('phenotype_values[]')

        logger.info(f"Taxonomy level: {taxonomy_level}, value: {taxonomy_value}")
        logger.info(f"Phenotype filters: {list(zip(phenotype_ids, phenotype_values))}")

        query = Q()
        if taxonomy_level and taxonomy_value:
            field_name = f'taxonomy__{taxonomy_level}'
            query &= Q(**{field_name: taxonomy_value})

        microbes = Microbe.objects.filter(query).distinct()

        if phenotype_ids and phenotype_values:
            for phenotype_id, phenotype_value in zip(phenotype_ids, phenotype_values):
                gt_match = Q(
                    Q(phenotype__definition__id=phenotype_id) &
                    Q(phenotype__value__icontains=phenotype_value)
                )
                pred_match = Q(
                    Q(predictions__predicted_phenotypes__definition__id=phenotype_id) &
                    Q(predictions__predicted_phenotypes__value__icontains=phenotype_value)
                )
                microbes = microbes.filter(gt_match | pred_match).distinct()

        for microbe in microbes:
            microbe.non_na_fields = microbe.phenotype_set.exclude(value__in=[None, '', 'N/A']).count()
            microbe.predictions_count = microbe.predictions.count()

        results = microbes

    context = {
        'results': results,
        'phenotype_definitions': phenotype_definitions_json,
    }
    return render(request, 'jsonl_viewer/search.html', context)


def browse_microbes(request):
    # Aggregate statistics
    total_species_with_ground_truth = Microbe.objects.filter(
        phenotype__value__isnull=False
    ).distinct().count()
    total_species_with_predictions = Microbe.objects.filter(
        predictions__isnull=False
    ).distinct().count()
    total_phenotypes = PhenotypeDefinition.objects.count()
    total_models = Prediction.objects.values('model').distinct().count()

    # Initialize the microbes queryset
    microbes_list = Microbe.objects.all()

    # Pagination
    paginator = Paginator(microbes_list, 20)  # Show 20 microbes per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Get the IDs of the microbes on the current page
    microbe_ids = [microbe.id for microbe in page_obj.object_list]

    # Now get a QuerySet of these microbes with prefetch_related
    microbes = Microbe.objects.filter(id__in=microbe_ids).prefetch_related(
        Prefetch(
            'phenotype_set',
            queryset=Phenotype.objects.exclude(value__in=[None, '', 'N/A']).select_related('definition')
        ),
        Prefetch(
            'predictions__predicted_phenotypes',
            queryset=PredictedPhenotype.objects.exclude(value__in=[None, '', 'N/A']).select_related('definition')
        ),
        Prefetch('descriptions')  # Prefetch MicrobeDescription
    )

    phenotype_definitions = list(PhenotypeDefinition.objects.all())

    # Build a dict of microbes by id for easy access
    microbe_dict = {microbe.id: microbe for microbe in microbes}

    # Process each microbe and add required attributes
    for microbe in microbes:
        # Ground truth phenotypes
        gt_phenotypes = {p.definition.id: p for p in microbe.phenotype_set.all()}
        microbe.ground_truth_count = len(gt_phenotypes)

        # Get prediction summaries
        prediction_summaries = {
            ps.definition_id: ps 
            for ps in PhenotypeSummary.objects.filter(microbe=microbe).select_related('definition')
        }
        
        # Count additional predictions (predictions without ground truth)
        microbe.additional_predictions_count = len(
            set(prediction_summaries.keys()) - set(gt_phenotypes.keys())
        )

        # Phenotypes availability
        phenotypes_available = []
        for pd in phenotype_definitions:
            if pd.name == "Member of WA subset":
                continue
            has_gt = pd.id in gt_phenotypes
            has_prediction = pd.id in prediction_summaries and prediction_summaries[pd.id].majority_value not in [None, '', 'N/A']
            phenotypes_available.append({
                'definition': pd,
                'has_gt': has_gt,
                'has_prediction': has_prediction,
            })
        microbe.phenotypes_available = phenotypes_available

    # Ensure the microbes are in the same order as in the page_obj
    processed_microbes = [microbe_dict[microbe_id] for microbe_id in microbe_ids]

    # Replace page_obj.object_list with the processed microbes
    page_obj.object_list = processed_microbes

    context = {
        'page_obj': page_obj,
        'phenotype_definitions': phenotype_definitions,
        'total_species_with_ground_truth': total_species_with_ground_truth,
        'total_species_with_predictions': total_species_with_predictions,
        'total_phenotypes': total_phenotypes,
        'total_models': total_models,
    }

    return render(request, 'jsonl_viewer/browse.html', context)

def report_error(request):
    if request.method == 'POST':
        description = request.POST.get('description', '').strip()
        microbe_id = request.POST.get('microbe_id')
        current_page = request.POST.get('current_page', '')

        if not description:
            messages.error(request, "Error description cannot be empty.")
            return redirect(current_page or 'index')

        try:
            microbe = Microbe.objects.get(id=microbe_id)
        except Microbe.DoesNotExist:
            microbe = None

        ErrorReport.objects.create(
            microbe=microbe,
            description=description
        )

        messages.success(request, "Thank you for reporting the error. Our team will review it.")
        return redirect(current_page or 'index')
    else:
        # For GET requests, redirect to home or appropriate page
        return redirect('index')

# ----- User Authentication Views -----

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful.")
            return redirect('home')
        else:
            messages.error(request, "Unsuccessful registration. Invalid information.")
    else:
        form = CustomUserCreationForm()
    return render(request, 'jsonl_viewer/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'jsonl_viewer/login.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('index')

# ----- Home View for Logged-in Users -----
@login_required
def home(request):
    # Fetch starred microbes
    starred_microbes = request.user.profile.starred_microbes.all()
    
    starred_microbes_with_changes = []
    
    for microbe in starred_microbes:
        # Get the most recent change with non-empty change_description
        recent_change = MicrobeDescription.objects.filter(
            microbe=microbe,
            change_description__isnull=False,  # Ensure change_description exists
            change_description__gt=''  # Ensure it's not empty
        ).order_by('-inference_date_time').first()
        
        starred_microbes_with_changes.append({
            'microbe': microbe,
            'recent_change': recent_change
        })

    context = {
        'starred_microbes_with_changes': starred_microbes_with_changes,
    }

    return render(request, 'jsonl_viewer/home.html', context)

@login_required  # Optional: Restrict access to logged-in users
def download_database(request):
    # Create the HttpResponse object with CSV headers.
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="database_export.csv"'

    writer = csv.writer(response)

    # Write Taxonomy Data
    writer.writerow(['Taxonomy ID', 'Superkingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
    for taxonomy in Taxonomy.objects.all():
        writer.writerow([
            taxonomy.id,
            taxonomy.superkingdom,
            taxonomy.phylum,
            taxonomy.class_name,
            taxonomy.order,
            taxonomy.family,
            taxonomy.genus,
            taxonomy.species
        ])

    # Write Microbe Data
    writer.writerow([])
    writer.writerow(['Microbe ID', 'Binomial Name', 'NCBI ID', 'Taxonomy ID', 'Alternative Names', 'FTP Path', 'Fasta File', 'Access Count Users', 'Access Count Tools'])
    for microbe in Microbe.objects.all():
        writer.writerow([
            microbe.id,
            microbe.binomial_name,
            microbe.ncbi_id,
            microbe.taxonomy.id,
            ', '.join(microbe.alternative_names),
            microbe.ftp_path,
            microbe.fasta_file,
            microbe.access_count_users,
            microbe.access_count_tools
        ])

    # Write PhenotypeDefinition Data
    writer.writerow([])
    writer.writerow(['PhenotypeDefinition ID', 'Name', 'Data Type', 'Allowed Values', 'Description'])
    for pd in PhenotypeDefinition.objects.all():
        writer.writerow([
            pd.id,
            pd.name,
            pd.data_type,
            ', '.join(pd.allowed_values_list),
            pd.description
        ])

    # Write Phenotype Data
    writer.writerow([])
    writer.writerow(['Phenotype ID', 'Microbe ID', 'Definition ID', 'Value'])
    for phenotype in Phenotype.objects.all():
        writer.writerow([
            phenotype.id,
            phenotype.microbe.id,
            phenotype.definition.id,
            json.dumps(phenotype.value)  # Serialize JSONField
        ])

    # Write Prediction Data
    writer.writerow([])
    writer.writerow(['Prediction ID', 'Microbe ID', 'Prediction Identifier', 'Model', 'Inference DateTime'])
    for prediction in Prediction.objects.all():
        writer.writerow([
            prediction.id,
            prediction.microbe.id,
            prediction.prediction_id,
            prediction.model,
            prediction.inference_date_time.strftime('%Y-%m-%d %H:%M:%S')
        ])

    # Write PredictedPhenotype Data
    writer.writerow([])
    writer.writerow(['PredictedPhenotype ID', 'Prediction ID', 'Definition ID', 'Value'])
    for pp in PredictedPhenotype.objects.all():
        writer.writerow([
            pp.id,
            pp.prediction.id,
            pp.definition.id,
            json.dumps(pp.value)  # Serialize JSONField
        ])

    # Write ModelRanking Data
    writer.writerow([])
    writer.writerow(['ModelRanking ID', 'Model', 'Target', 'Balanced Accuracy', 'Precision', 'Sample Size'])
    for mr in ModelRanking.objects.all():
        writer.writerow([
            mr.id,
            mr.model,
            mr.target,
            mr.balanced_accuracy,
            mr.precision,
            mr.sample_size
        ])

    # Write MicrobeDescription Data
    writer.writerow([])
    writer.writerow(['MicrobeDescription ID', 'Microbe ID', 'Description Type', 'Model', 'Description', 'Inference DateTime'])
    for md in MicrobeDescription.objects.all():
        writer.writerow([
            md.id,
            md.microbe.id,
            md.description_type,
            md.model,
            md.description,
            md.inference_date_time.strftime('%Y-%m-%d %H:%M:%S')
        ])

    # Write ErrorReport Data
    writer.writerow([])
    writer.writerow(['ErrorReport ID', 'Microbe ID', 'Description', 'Created At'])
    for er in ErrorReport.objects.all():
        writer.writerow([
            er.id,
            er.microbe.id,
            er.description,
            er.created_at.strftime('%Y-%m-%d %H:%M:%S')
        ])

    return response