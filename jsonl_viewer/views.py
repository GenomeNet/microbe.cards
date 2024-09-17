import os
import json
from django.conf import settings
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.core.paginator import Paginator
from django.db.models import Count, Prefetch, Q
from collections import defaultdict, OrderedDict
from .models import Microbe, Phenotype, PhenotypeDefinition, Prediction, PredictedPhenotype, ModelRanking, Taxonomy
import logging
from django.template.defaultfilters import register
from urllib.parse import unquote

logger = logging.getLogger(__name__)

@register.filter
def subtract(value, arg):
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

def index(request):
    total_species_with_ground_truth = Microbe.objects.filter(
        phenotype__value__isnull=False
    ).distinct().count()
    total_species_with_predictions = Microbe.objects.filter(
        predictions__isnull=False
    ).distinct().count()
    total_phenotypes = PhenotypeDefinition.objects.count()
    total_models = Prediction.objects.values('model').distinct().count()

    search_results = None
    phenotype_definitions = PhenotypeDefinition.objects.all()

    # Get parameters from the request
    include_no_predictions = request.POST.get('include_no_predictions', 'false') == 'true'
    selected_phenotype = request.POST.get('selected_phenotype', '')
    selected_value = request.POST.get('selected_value', '')
    search_term = request.POST.get('search_term', '').strip()

    microbes = Microbe.objects.all()

    if not include_no_predictions:
        microbes = microbes.filter(predictions__isnull=False).distinct()

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
                    Q(predictions__predicted_phenotypes__definition=phenotype_def,
                      predictions__predicted_phenotypes__value__iexact=selected_value)
                ).distinct()
            else:
                microbes = microbes.filter(
                    predictions__predicted_phenotypes__definition=phenotype_def
                ).distinct()

    if request.method == 'POST':
        # Optimize by prefetching related data
        microbes = microbes.prefetch_related(
            Prefetch('phenotype_set', queryset=Phenotype.objects.exclude(value__in=[None, '', 'N/A']).select_related('definition')),
            Prefetch('predictions__predicted_phenotypes', queryset=PredictedPhenotype.objects.exclude(value__in=[None, '', 'N/A']).select_related('definition'))
        )

        phenotype_definitions_all = list(phenotype_definitions)

        for microbe in microbes:
            # Ground truth phenotypes
            gt_phenotypes = {p.definition.id: p for p in microbe.phenotype_set.all()}
            microbe.ground_truth_count = len(gt_phenotypes)

            # Predicted phenotypes
            predicted_phenotypes = {}
            for prediction in microbe.predictions.all():
                for pp in prediction.predicted_phenotypes.all():
                    predicted_phenotypes[pp.definition.id] = pp
            microbe.additional_predictions_count = len(set(predicted_phenotypes.keys()) - set(gt_phenotypes.keys()))

            # Phenotypes availability
            phenotypes_available = []
            for pd in phenotype_definitions_all:
                has_gt = pd.id in gt_phenotypes
                has_prediction = pd.id in predicted_phenotypes
                phenotypes_available.append({
                    'definition': pd,
                    'has_gt': has_gt,
                    'has_prediction': has_prediction,
                })
            microbe.phenotypes_available = phenotypes_available

        search_results = microbes

    context = {
        'total_species_with_ground_truth': total_species_with_ground_truth,
        'total_species_with_predictions': total_species_with_predictions,
        'total_phenotypes': total_phenotypes,
        'total_models': total_models,
        'search_results': search_results,
        'phenotype_definitions': phenotype_definitions,
        'include_no_predictions': include_no_predictions,
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
    microbe = get_object_or_404(Microbe, id=microbe_id)
    phenotype_definitions = PhenotypeDefinition.objects.all()
    microbe_phenotypes = Phenotype.objects.filter(microbe=microbe).select_related('definition')
    microbe_phenotypes_dict = {p.definition.id: p for p in microbe_phenotypes}

    predictions = Prediction.objects.filter(microbe=microbe).prefetch_related(
        'predicted_phenotypes__definition'
    ).order_by('-inference_date_time')

    models_used = set(prediction.model for prediction in predictions)
    predicted_phenotype_definitions = set()
    for prediction in predictions:
        for predicted_phenotype in prediction.predicted_phenotypes.all():
            predicted_phenotype_definitions.add(predicted_phenotype.definition.name)

    model_rankings = ModelRanking.objects.filter(
        model__in=models_used,
        target__in=predicted_phenotype_definitions
    )

    model_ranking_dict = {}
    for mr in model_rankings:
        model_ranking_dict.setdefault(mr.model, {})[mr.target] = mr

    phenotype_data = []
    ground_truth_count = 0

    # Initialize a dictionary to count 'N/A' per model
    na_counts = defaultdict(int)

    for phenotype_def in phenotype_definitions:
        if phenotype_def.name == "Member of WA subset":
            continue
        phenotype = microbe_phenotypes_dict.get(phenotype_def.id)
        ground_truth_value = phenotype.value if phenotype else None
        if ground_truth_value not in [None, '', 'N/A']:
            ground_truth_count += 1
        data = {
            'phenotype_definition': phenotype_def,
            'ground_truth': ground_truth_value,
            'predictions': []
        }
        for prediction in predictions:
            predicted_value = None
            model_performance = model_ranking_dict.get(prediction.model, {}).get(phenotype_def.name)
            predicted_phenotype = prediction.predicted_phenotypes.filter(definition=phenotype_def).first()
            if predicted_phenotype:
                predicted_value = predicted_phenotype.value
            else:
                # Increment 'N/A' count if no prediction is available
                na_counts[prediction.model] += 1
            data['predictions'].append({
                'model': prediction.model,
                'predicted_value': predicted_value,
                'model_performance': model_performance,
                'inference_date': prediction.inference_date_time,
            })
        phenotype_data.append(data)

    # Sort the models based on fewest 'N/A' counts (ascending)
    sorted_models = sorted(models_used, key=lambda m: na_counts.get(m, 0))

    # Create an ordered list of predictions based on sorted models
    sorted_predictions = sorted(
        predictions,
        key=lambda p: sorted_models.index(p.model) if p.model in sorted_models else len(sorted_models)
    )

    # Update phenotype_data to reflect the sorted predictions
    phenotype_data_sorted = []
    for data in phenotype_data:
        sorted_pred = sorted(
            data['predictions'],
            key=lambda p: sorted_models.index(p['model']) if p['model'] in sorted_models else len(sorted_models)
        )
        data['predictions'] = sorted_pred
        phenotype_data_sorted.append(data)

    # Count non-'N/A' predictions per model for context (optional)
    model_prediction_counts = {model: 0 for model in models_used}
    for data in phenotype_data_sorted:
        for pred in data['predictions']:
            if pred['predicted_value'] not in [None, '', 'N/A']:
                model_prediction_counts[pred['model']] += 1

    additional_predictions_count = 0
    for data in phenotype_data_sorted:
        if data['ground_truth'] in [None, '', 'N/A']:
            has_prediction = any(pred['predicted_value'] not in [None, '', 'N/A'] for pred in data['predictions'])
            if has_prediction:
                additional_predictions_count += 1

    context = {
        'microbe': microbe,
        'phenotype_data': phenotype_data_sorted,  # Use the sorted phenotype data
        'ground_truth_count': ground_truth_count,
        'model_prediction_counts': model_prediction_counts,
        'additional_predictions_count': additional_predictions_count,
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

import logging

logger = logging.getLogger(__name__)

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

def about(request):
    return render(request, 'jsonl_viewer/about.html')

def imprint(request):
    return render(request, 'jsonl_viewer/imprint.html')

def search(request):
    logger.info("Search view called")
    results = None
    phenotype_definitions = PhenotypeDefinition.objects.all()
    phenotype_definitions_json = [
        {
            'id': pd.id,
            'text': pd.name,
            'allowed_values': pd.allowed_values_list,
        } for pd in phenotype_definitions
    ]
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