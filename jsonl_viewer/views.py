import os
import json
from django.conf import settings
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.core.paginator import Paginator
from django.db.models import Count, Prefetch
from collections import defaultdict
from .models import Microbe, Phenotype, PhenotypeDefinition, Prediction, PredictedPhenotype  # Update this line

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
    phenotypes = Phenotype.objects.filter(microbe=microbe).select_related('definition')
    predictions = Prediction.objects.filter(microbe=microbe).prefetch_related(
        'predicted_phenotypes__definition'
    ).order_by('-inference_date_time')
    
    for phenotype in phenotypes:
        if phenotype.definition.allowed_values:
            phenotype.definition.allowed_values = phenotype.definition.get_allowed_values()
    
    context = {
        'microbe': microbe,
        'phenotypes': phenotypes,
        'predictions': predictions,
    }
    return render(request, 'jsonl_viewer/card.html', context)
