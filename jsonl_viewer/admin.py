# jsonl_viewer/admin.py

from django.contrib import admin
from .models import (
    Taxonomy,
    Microbe,
    PhenotypeDefinition,
    Phenotype,
    Prediction,
    PredictedPhenotype,
    ModelRanking,
    MicrobeDescription,
    ErrorReport
)

# Register Taxonomy Model
@admin.register(Taxonomy)
class TaxonomyAdmin(admin.ModelAdmin):
    list_display = ('superkingdom', 'phylum', 'class_name', 'order', 'family', 'genus', 'species')
    search_fields = ('superkingdom', 'phylum', 'class_name', 'order', 'family', 'genus', 'species')

# Register Microbe Model
@admin.register(Microbe)
class MicrobeAdmin(admin.ModelAdmin):
    list_display = ('binomial_name', 'ncbi_id', 'taxonomy', 'ftp_path', 'fasta_file')
    search_fields = ('binomial_name', 'ftp_path', 'fasta_file')
    list_filter = ('taxonomy',)

# Register PhenotypeDefinition Model
@admin.register(PhenotypeDefinition)
class PhenotypeDefinitionAdmin(admin.ModelAdmin):
    list_display = ('name', 'data_type')
    search_fields = ('name', 'description')
    list_filter = ('data_type',)

# Register Phenotype Model
@admin.register(Phenotype)
class PhenotypeAdmin(admin.ModelAdmin):
    list_display = ('microbe', 'definition', 'value')
    search_fields = ('microbe__binomial_name', 'definition__name')
    list_filter = ('definition',)

# Register Prediction Model
@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('microbe', 'prediction_id', 'model', 'inference_date_time')
    search_fields = ('microbe__binomial_name', 'prediction_id', 'model')
    list_filter = ('model', 'inference_date_time')

# Register PredictedPhenotype Model
@admin.register(PredictedPhenotype)
class PredictedPhenotypeAdmin(admin.ModelAdmin):
    list_display = ('prediction', 'definition', 'value')
    search_fields = ('prediction__prediction_id', 'definition__name')
    list_filter = ('definition',)

# Register ModelRanking Model
@admin.register(ModelRanking)
class ModelRankingAdmin(admin.ModelAdmin):
    list_display = ('model', 'target', 'balanced_accuracy', 'precision', 'sample_size')
    search_fields = ('model', 'target')
    list_filter = ('model', 'target')

# Register MicrobeDescription Model
@admin.register(MicrobeDescription)
class MicrobeDescriptionAdmin(admin.ModelAdmin):
    list_display = ('microbe', 'description_type', 'model', 'inference_date_time')
    search_fields = ('microbe__binomial_name', 'description_type', 'model')
    list_filter = ('description_type', 'model')

# Register ErrorReport Model
@admin.register(ErrorReport)
class ErrorReportAdmin(admin.ModelAdmin):
    list_display = ('microbe', 'description', 'created_at')
    search_fields = ('microbe__binomial_name', 'description')
    list_filter = ('created_at',)
