from django.db import models
import json
from django.contrib.auth.models import User

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


class PhenotypeSummary(models.Model):
    microbe = models.ForeignKey(Microbe, on_delete=models.CASCADE, related_name='phenotype_summaries')
    definition = models.ForeignKey(PhenotypeDefinition, on_delete=models.CASCADE)
    majority_value = models.CharField(max_length=255, default='N/A')
    agreement_percentage = models.PositiveIntegerField(default=0)
    supporting_models = models.PositiveIntegerField(default=0)
    total_models = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ('microbe', 'definition')
        indexes = [
            models.Index(fields=['microbe', 'definition']),
            models.Index(fields=['majority_value']),
        ]

    def __str__(self):
        return f"{self.microbe.binomial_name} - {self.definition.name} Summary"
    
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

# Signal to create or update Profile whenever User is created or updated
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    instance.profile.save()


