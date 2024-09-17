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
    binomial_name = models.CharField(max_length=255, unique=True)  # Add unique=True
    ncbi_id = models.IntegerField(null=True, blank=True)
    taxonomy = models.ForeignKey(Taxonomy, on_delete=models.CASCADE)
    alternative_names = models.JSONField(default=list)
    ftp_path = models.CharField(max_length=500, null=True, blank=True)
    fasta_file = models.CharField(max_length=200, null=True, blank=True)

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

    class Meta:
        unique_together = ('microbe', 'description_type', 'model')

    def __str__(self):
        return f"{self.microbe.binomial_name} - {self.description_type} ({self.model})"