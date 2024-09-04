import os
import json
from django.conf import settings
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from .models import GroundTruthData

def index(request):
    entries = GroundTruthData.objects.all().select_related()
    entries_by_taxonomy = {}

    for entry in entries:
        superkingdom = entry.superkingdom
        phylum = entry.phylum
        class_field = entry.class_field
        order = entry.order
        family = entry.family
        genus = entry.genus

        # Calculate the number of non-NA fields
        non_na_fields = sum(1 for field in [
            entry.ncbi_id, entry.motility, entry.gram_staining, entry.aerophilicity,
            entry.extreme_environment_tolerance, entry.biofilm_formation, entry.animal_pathogenicity,
            entry.biosafety_level, entry.health_association, entry.host_association,
            entry.plant_pathogenicity, entry.spore_formation, entry.hemolysis,
            entry.cell_shape, entry.member_of_wa_subset, entry.ftp_path, entry.fasta_file
        ] if field and field != 'NA')
        total_fields = 17
        missing_fields = total_fields - non_na_fields

        entry_info = {
            'entry': entry,
            'non_na_fields': non_na_fields,
            'missing_fields': missing_fields
        }

        if superkingdom not in entries_by_taxonomy:
            entries_by_taxonomy[superkingdom] = {}
        if phylum not in entries_by_taxonomy[superkingdom]:
            entries_by_taxonomy[superkingdom][phylum] = {}
        if class_field not in entries_by_taxonomy[superkingdom][phylum]:
            entries_by_taxonomy[superkingdom][phylum][class_field] = {}
        if order not in entries_by_taxonomy[superkingdom][phylum][class_field]:
            entries_by_taxonomy[superkingdom][phylum][class_field][order] = {}
        if family not in entries_by_taxonomy[superkingdom][phylum][class_field][order]:
            entries_by_taxonomy[superkingdom][phylum][class_field][order][family] = {}
        if genus not in entries_by_taxonomy[superkingdom][phylum][class_field][order][family]:
            entries_by_taxonomy[superkingdom][phylum][class_field][order][family][genus] = []

        entries_by_taxonomy[superkingdom][phylum][class_field][order][family][genus].append(entry_info)

    return render(request, 'jsonl_viewer/index.html', {'entries_by_taxonomy': entries_by_taxonomy})

def entry_detail(request, species):
    entry = get_object_or_404(GroundTruthData, species=species)
    json_data = {
        'NCBI ID': entry.ncbi_id,
        'Motility': entry.motility,
        'Gram Staining': entry.gram_staining,
        'Aerophilicity': entry.aerophilicity,
        'Extreme Environment Tolerance': entry.extreme_environment_tolerance,
        'Biofilm Formation': entry.biofilm_formation,
        'Animal Pathogenicity': entry.animal_pathogenicity,
        'Biosafety Level': entry.biosafety_level,
        'Health Association': entry.health_association,
        'Host Association': entry.host_association,
        'Plant Pathogenicity': entry.plant_pathogenicity,
        'Spore Formation': entry.spore_formation,
        'Hemolysis': entry.hemolysis,
        'Cell Shape': entry.cell_shape,
        'Member of WA Subset': entry.member_of_wa_subset,
        'Superkingdom': entry.superkingdom,
        'Phylum': entry.phylum,
        'Class': entry.class_field,
        'Order': entry.order,
        'Family': entry.family,
        'Genus': entry.genus,
        'Species': entry.species,
        'FTP Path': entry.ftp_path,
        'Fasta File': entry.fasta_file,
    }
    return render(request, 'jsonl_viewer/entry_detail.html', {'entry': entry, 'json_data': json_data})