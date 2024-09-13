from django.db import models
import json

class Taxonomy(models.Model):
    superkingdom = models.CharField(max_length=100)
    phylum = models.CharField(max_length=100)
    class_name = models.CharField(max_length=100)
    order = models.CharField(max_length=100)
    family = models.CharField(max_length=100)
    genus = models.CharField(max_length=100)
    species = models.CharField(max_length=100)

class Microbe(models.Model):
    binomial_name = models.CharField(max_length=255)
    ncbi_id = models.IntegerField(null=True, blank=True)  # Allow null values
    taxonomy = models.ForeignKey(Taxonomy, on_delete=models.CASCADE)
    alternative_names = models.JSONField(default=list)
    ftp_path = models.CharField(max_length=500, null=True, blank=True)
    fasta_file = models.CharField(max_length=200, null=True, blank=True)

class PhenotypeDefinition(models.Model):
    name = models.CharField(max_length=100, unique=True)
    data_type = models.CharField(max_length=50)
    allowed_values = models.JSONField(default=list)
    description = models.TextField()

    def set_allowed_values(self, values):
        self.allowed_values = json.dumps(values)

    def get_allowed_values(self):
        return json.loads(self.allowed_values)

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