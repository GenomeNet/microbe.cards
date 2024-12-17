from django.db import models
import json
from django.contrib.auth.models import User  # Import User model
from django.utils import timezone
from django.contrib.postgres.fields import JSONField  # If using Postgres


class Taxonomy(models.Model):
    superkingdom = models.CharField(max_length=100)
    phylum = models.CharField(max_length=100)
    class_name = models.CharField(max_length=100)
    order = models.CharField(max_length=100)
    family = models.CharField(max_length=100)
    genus = models.CharField(max_length=100)
    species = models.CharField(max_length=100)

class Microbe(models.Model):
    binomial_name = models.CharField(max_length=255, unique=True)
    ncbi_id = models.IntegerField(null=True, blank=True)
    taxonomy = models.ForeignKey(Taxonomy, on_delete=models.CASCADE)
    alternative_names = models.JSONField(default=list)
    ftp_path = models.CharField(max_length=500, null=True, blank=True)
    fasta_file = models.CharField(max_length=200, null=True, blank=True)
    
    access_count_users = models.PositiveIntegerField(default=0)
    access_count_tools = models.PositiveIntegerField(default=0)

    h_index = models.IntegerField(null=True, blank=True, db_index=True)
    total_citations = models.IntegerField(null=True, blank=True, db_index=True)
    recent_citations_3yr = models.IntegerField(null=True, blank=True, db_index=True)

    def __str__(self):
        return self.binomial_name

class PhenotypeDefinition(models.Model):
    name = models.CharField(max_length=100, unique=True)
    data_type = models.CharField(max_length=50)
    allowed_values = models.JSONField(default=list)
    description = models.TextField()

    @property
    def allowed_values_list(self):
        if isinstance(self.allowed_values, list):
            return self.allowed_values
        elif isinstance(self.allowed_values, str):
            try:
                data = json.loads(self.allowed_values)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
            except json.JSONDecodeError:
                return [self.allowed_values]
        else:
            return []

class Phenotype(models.Model):
    microbe = models.ForeignKey(Microbe, on_delete=models.CASCADE)
    definition = models.ForeignKey(PhenotypeDefinition, on_delete=models.CASCADE)
    value = models.JSONField()

class Prediction(models.Model):
    microbe = models.ForeignKey(Microbe, on_delete=models.CASCADE, related_name='predictions')
    prediction_id = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    inference_date_time = models.DateTimeField()

    class Meta:
        unique_together = ('microbe', 'prediction_id', 'inference_date_time', 'model')

class PhenotypeStats(models.Model):
    phenotype_definition = models.OneToOneField(PhenotypeDefinition, on_delete=models.CASCADE, related_name='stats')
    ground_truth_count = models.IntegerField(default=0)
    predicted_count = models.IntegerField(default=0)
    total_microbes = models.IntegerField(default=0)
    value_distribution = models.JSONField(default=dict)  # Django 3.1+ has models.JSONField
    
    def __str__(self):
        return f"Stats for {self.phenotype_definition.name}"

class PredictedPhenotype(models.Model):
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, related_name='predicted_phenotypes')
    definition = models.ForeignKey(PhenotypeDefinition, on_delete=models.CASCADE)
    value = models.JSONField()

class ModelRanking(models.Model):
    model = models.CharField(max_length=255)
    target = models.CharField(max_length=255)
    balanced_accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    sample_size = models.IntegerField()

    class Meta:
        unique_together = ('model', 'target')

    def __str__(self):
        return f"{self.model} - {self.target}"

class MicrobeDescription(models.Model):
    microbe = models.ForeignKey(Microbe, on_delete=models.CASCADE, related_name='descriptions')
    description_type = models.CharField(max_length=50)
    model = models.CharField(max_length=100)
    description = models.TextField()
    inference_date_time = models.DateTimeField(auto_now=True)
    hidden = models.BooleanField(default=False)
    change_description = models.TextField(blank=True)

    class Meta:
        unique_together = ('microbe', 'description_type', 'model')

    def __str__(self):
        return f"{self.microbe.binomial_name} - {self.description_type} ({self.model})"

class MicrobeRequest(models.Model):
    STATUS_CHOICES = [
        ('under_review', 'Under Review'),
        ('processing', 'Processing'),
        ('published', 'Published'),
    ]

    binomial_name = models.CharField(max_length=255, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='microbe_requests')
    institute = models.CharField(max_length=255, null=True, blank=True)
    email = models.EmailField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='under_review')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    comments = models.TextField(blank=True)  # Optional field for admin comments

    def __str__(self):
        return f"Request: {self.binomial_name} by {self.user.username} - {self.get_status_display()}"

class ErrorReport(models.Model):
    microbe = models.ForeignKey(Microbe, on_delete=models.CASCADE, related_name='error_reports')
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Error Report for {self.microbe.binomial_name} at {self.created_at}"

class MicrobeAccess(models.Model):
    microbe = models.ForeignKey(Microbe, on_delete=models.CASCADE, related_name='accesses')
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)  # Associated user if logged in
    accessed_at = models.DateTimeField(auto_now_add=True)
    search_tool = models.CharField(max_length=255, null=True, blank=True)  # Identifier for search tools

    def __str__(self):
        if self.user:
            return f"Access by {self.user.username} to {self.microbe.binomial_name} at {self.accessed_at}"
        elif self.search_tool:
            return f"Access by {self.search_tool} to {self.microbe.binomial_name} at {self.accessed_at}"
        else:
            return f"Access to {self.microbe.binomial_name} at {self.accessed_at}"

# Add the following Profile model below existing models

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    starred_microbes = models.ManyToManyField(Microbe, blank=True, related_name='starred_by')
    institute = models.CharField(max_length=255, null=True, blank=True)  # New field for institute

    def __str__(self):
        return f"{self.user.username}'s Profile"


class Manuscript(models.Model):
    microbe = models.ForeignKey(Microbe, on_delete=models.CASCADE, related_name='manuscripts')
    title = models.TextField()
    abstract = models.TextField(blank=True)
    authors = models.JSONField(default=list, blank=True)
    pmcid = models.CharField(max_length=50, null=True, blank=True, db_index=True)
    source = models.CharField(max_length=50, blank=True)
    journal_title = models.CharField(max_length=255, blank=True)
    is_open_access = models.BooleanField(default=False)
    published_year = models.IntegerField(null=True, blank=True)
    published_date = models.DateField(null=True, blank=True)
    citation_count = models.IntegerField(null=True, blank=True)
    citation_updated_at = models.DateTimeField(null=True, blank=True)   
    

    class Meta:
        unique_together = ('microbe', 'title')
        
    def __str__(self):
        return f"{self.title[:50]}... ({self.microbe.binomial_name})"

class CitingManuscript(models.Model):
    cited_manuscript = models.ForeignKey(Manuscript, on_delete=models.CASCADE, related_name='citations')
    pmid = models.CharField(max_length=50, null=True, blank=True, db_index=True)
    pmcid = models.CharField(max_length=50, null=True, blank=True, db_index=True)
    title = models.TextField(blank=True)
    abstract = models.TextField(blank=True)
    authors = models.JSONField(default=list, blank=True)
    journal_title = models.CharField(max_length=255, blank=True)
    published_date = models.DateField(null=True, blank=True)

    class Meta:
        unique_together = ('cited_manuscript', 'pmid', 'pmcid')
    
    def __str__(self):
        return f"Citing article {self.title[:50]}... for {self.cited_manuscript.microbe.binomial_name}"
    
# Signal to create or update Profile whenever User is created or updated
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    instance.profile.save()




# New Model for Feedback
class SectionFeedback(models.Model):
    microbe = models.ForeignKey(Microbe, on_delete=models.CASCADE, related_name='feedbacks')
    section_name = models.CharField(max_length=255)
    rating = models.CharField(max_length=10)  # 'up' or 'down'
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Feedback on {self.microbe.binomial_name} - {self.section_name}: {self.rating}"
    

class Genome(models.Model):
    microbe = models.OneToOneField(Microbe, on_delete=models.CASCADE, related_name='genome')
    assembly_accession = models.CharField(max_length=100, unique=True)
    genome_size = models.IntegerField(blank=True, null=True)
    gc_content = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"Genome of {self.microbe.binomial_name}"
    
class Feature(models.Model):
    genome = models.ForeignKey(Genome, on_delete=models.CASCADE, related_name='features')
    feature_type = models.CharField(max_length=50)
    seqid = models.CharField(max_length=255, blank=True, null=True)
    source = models.CharField(max_length=255, blank=True, null=True)
    start = models.IntegerField()
    end = models.IntegerField()
    strand = models.CharField(
        max_length=1, 
        choices=[('+', 'Forward'), ('-', 'Reverse')], 
        blank=True, 
        null=True
    )
    attributes = models.JSONField(default=dict, blank=True)

    # New fields:
    # methods: a JSONField to store one or multiple methods/tools with their versions and parameters
    # Example: {"prokka": {"version": "1.14.6", "date": "2022-03-10"},"toolX":{"version":"2.0"}}
    methods = models.JSONField(default=dict, blank=True)

    # If you want to store the time when this feature was first imported
    imported_at = models.DateTimeField(default=timezone.now)

    parent = models.ForeignKey(
        'self', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='child_features'
    )

    def __str__(self):
        return f"{self.feature_type} ({self.start}-{self.end}) [{self.genome.assembly_accession}]"

    @property
    def locus_tag(self):
        return self.attributes.get('locus_tag')

    @property
    def product(self):
        return self.attributes.get('product')

    @property
    def gene_name(self):
        return self.attributes.get('gene')