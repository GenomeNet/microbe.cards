import os
import json
from django.conf import settings
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.core.paginator import Paginator
from django.db.models import Count, Prefetch, Q
from collections import defaultdict, OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from .models import Microbe, Phenotype, PhenotypeDefinition, Prediction, PredictedPhenotype, ModelRanking, Taxonomy
import numpy as np
import logging
from django.template.defaultfilters import register

logger = logging.getLogger(__name__)

@register.filter
def subtract(value, arg):
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

def index(request):
    microbes = Microbe.objects.select_related('taxonomy').prefetch_related(
        Prefetch('phenotype_set', queryset=Phenotype.objects.select_related('definition'))
    )
    
    entries_by_taxonomy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))

    total_strains = 0
    total_species = set()
    total_families = set()
    total_classes = set()

    phenotype_definitions = PhenotypeDefinition.objects.all()
    total_phenotypes = phenotype_definitions.count()

    for microbe in microbes:
        taxonomy = microbe.taxonomy
        entries_by_taxonomy[taxonomy.superkingdom][taxonomy.phylum][taxonomy.class_name][taxonomy.order][taxonomy.family][taxonomy.genus].append(microbe)
        
        total_strains += 1
        total_species.add(microbe.binomial_name)
        total_families.add(taxonomy.family)
        total_classes.add(taxonomy.class_name)

        # Count non-NA and missing phenotypes
        phenotypes = microbe.phenotype_set.all()
        non_na_fields = sum(1 for p in phenotypes if p.value is not None and p.value != '')
        missing_fields = total_phenotypes - non_na_fields
        microbe.non_na_fields = non_na_fields
        microbe.missing_fields = missing_fields

    # Convert defaultdict to regular dict for template rendering
    entries_by_taxonomy = dict(entries_by_taxonomy)
    for superkingdom, phyla in entries_by_taxonomy.items():
        entries_by_taxonomy[superkingdom] = dict(phyla)
        for phylum, classes in phyla.items():
            entries_by_taxonomy[superkingdom][phylum] = dict(classes)
            for class_name, orders in classes.items():
                entries_by_taxonomy[superkingdom][phylum][class_name] = dict(orders)
                for order, families in orders.items():
                    entries_by_taxonomy[superkingdom][phylum][class_name][order] = dict(families)
                    for family, genera in families.items():
                        entries_by_taxonomy[superkingdom][phylum][class_name][order][family] = dict(genera)

    context = {
        'entries_by_taxonomy': entries_by_taxonomy,
        'total_strains': total_strains,
        'total_species': len(total_species),
        'total_families': len(total_families),
        'total_classes': len(total_classes),
    }

    return render(request, 'jsonl_viewer/index.html', context)

def microbe_detail(request, microbe_id):
    microbe = get_object_or_404(Microbe, id=microbe_id)
    # Fetch all phenotype definitions
    phenotype_definitions = PhenotypeDefinition.objects.all()
    # Fetch phenotypes for the microbe
    microbe_phenotypes = Phenotype.objects.filter(microbe=microbe).select_related('definition')
    # Create a dictionary for quick access
    microbe_phenotypes_dict = {p.definition.id: p for p in microbe_phenotypes}

    predictions = Prediction.objects.filter(microbe=microbe).prefetch_related(
        'predicted_phenotypes__definition'
    ).order_by('-inference_date_time')

    # Collect models and phenotypes used in predictions
    models_used = set(prediction.model for prediction in predictions)
    predicted_phenotype_definitions = set()
    for prediction in predictions:
        for predicted_phenotype in prediction.predicted_phenotypes.all():
            predicted_phenotype_definitions.add(predicted_phenotype.definition.name)

    # Fetch model performance data
    model_rankings = ModelRanking.objects.filter(
        model__in=models_used,
        target__in=predicted_phenotype_definitions
    )

    # Build a nested dict for easy access in templates
    model_ranking_dict = {}
    for mr in model_rankings:
        model_ranking_dict.setdefault(mr.model, {})[mr.target] = mr

    # Prepare phenotype data for the template
    phenotype_data = []
    for phenotype_def in phenotype_definitions:
        # Skip the phenotype "Member of WA subset" if needed
        if phenotype_def.name == "Member of WA subset":
            continue
        # Get the phenotype value for this microbe if it exists
        phenotype = microbe_phenotypes_dict.get(phenotype_def.id)
        ground_truth_value = phenotype.value if phenotype else None
        data = {
            'phenotype_definition': phenotype_def,
            'ground_truth': ground_truth_value,
            'predictions': []
        }
        for prediction in predictions:
            predicted_value = None
            model_performance = model_ranking_dict.get(prediction.model, {}).get(phenotype_def.name)
            # Find the predicted value for this phenotype definition
            predicted_phenotype = prediction.predicted_phenotypes.filter(definition=phenotype_def).first()
            if predicted_phenotype:
                predicted_value = predicted_phenotype.value
            data['predictions'].append({
                'model': prediction.model,
                'predicted_value': predicted_value,
                'model_performance': model_performance,
                'inference_date': prediction.inference_date_time,
            })
        phenotype_data.append(data)

    context = {
        'microbe': microbe,
        'phenotype_data': phenotype_data,
    }
    return render(request, 'jsonl_viewer/card.html', context)

def model_ranking(request):
    rankings = {}
    for ranking in ModelRanking.objects.all():
        if ranking.target not in rankings:
            rankings[ranking.target] = {}
        rankings[ranking.target][ranking.model] = {
            'balanced_accuracy': ranking.balanced_accuracy,
            'precision': ranking.precision,
            'sample_size': ranking.sample_size
        }

    # Sort the models for each phenotype by balanced accuracy
    sorted_rankings = {}
    for phenotype, models in rankings.items():
        sorted_rankings[phenotype] = OrderedDict(
            sorted(
                models.items(),
                key=lambda x: x[1]['balanced_accuracy'] if x[1]['balanced_accuracy'] is not None else -1,
                reverse=True
            )
        )

    return render(request, 'jsonl_viewer/model_ranking.html', {'rankings': sorted_rankings})

def about(request):
    return render(request, 'jsonl_viewer/about.html')

def imprint(request):
    return render(request, 'jsonl_viewer/imprint.html')

def search(request):
    logger.info("Search view called")
    results = None
    taxonomy_search = ''
    if request.method == 'POST':
        logger.info("POST request received")
        taxonomy_search = request.POST.get('taxonomy_search', '')
        taxonomy_rank = request.POST.get('taxonomy_rank', '')
        logger.info(f"Taxonomy search term: {taxonomy_search}, Rank: {taxonomy_rank}")
        
        if taxonomy_search:
            if taxonomy_rank == 'full':
                # Split the taxonomy string and get the most specific (last) part
                taxonomy_parts = taxonomy_search.split('>')
                specific_term = taxonomy_parts[-1].strip()
            else:
                # Remove the rank prefix (e.g., "Genus: ")
                specific_term = taxonomy_search.split(': ', 1)[-1].strip()
            
            logger.info(f"Searching for specific term: {specific_term}")
            
            query = Q()
            if taxonomy_rank == 'full' or taxonomy_rank == 'species':
                query |= Q(binomial_name__icontains=specific_term)
            
            rank_field = {
                'superkingdom': 'taxonomy__superkingdom',
                'phylum': 'taxonomy__phylum',
                'class': 'taxonomy__class_name',
                'order': 'taxonomy__order',
                'family': 'taxonomy__family',
                'genus': 'taxonomy__genus',
                'species': 'taxonomy__species'
            }.get(taxonomy_rank, '')
            
            if rank_field:
                query |= Q(**{f"{rank_field}__iexact": specific_term})
            else:
                # If no specific rank or 'full', search all fields
                query |= (
                    Q(taxonomy__superkingdom__icontains=specific_term) |
                    Q(taxonomy__phylum__icontains=specific_term) |
                    Q(taxonomy__class_name__icontains=specific_term) |
                    Q(taxonomy__order__icontains=specific_term) |
                    Q(taxonomy__family__icontains=specific_term) |
                    Q(taxonomy__genus__icontains=specific_term) |
                    Q(taxonomy__species__icontains=specific_term)
                )
            
            results = Microbe.objects.filter(query).distinct()
            logger.info(f"Query executed, {results.count()} results found")
        else:
            logger.warning("No taxonomy search term provided")
    else:
        logger.info("GET request received")
    
    context = {
        'results': results,
        'search_term': taxonomy_search if request.method == 'POST' else None
    }
    return render(request, 'jsonl_viewer/search.html', context)

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
        ).distinct()[:10]

        for tax in taxonomies:
            # Add full taxonomic path
            results.append({
                'id': f"full_{tax.id}",
                'text': f"{tax.superkingdom} > {tax.phylum} > {tax.class_name} > {tax.order} > {tax.family} > {tax.genus} > {tax.species}",
                'rank': 'full'
            })
            
            # Add individual ranks if they match the query
            ranks = [
                ('superkingdom', tax.superkingdom),
                ('phylum', tax.phylum),
                ('class', tax.class_name),
                ('order', tax.order),
                ('family', tax.family),
                ('genus', tax.genus),
                ('species', tax.species)
            ]
            
            for rank, value in ranks:
                if query.lower() in value.lower():
                    results.append({
                        'id': f"{rank}_{tax.id}",
                        'text': f"{rank.capitalize()}: {value}",
                        'rank': rank
                    })

    return JsonResponse({'results': results})
