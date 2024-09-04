from django.db import models
import json

class JSONLEntry(models.Model):
    custom_id = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    content = models.JSONField()

    def __str__(self):
        return self.name

    @classmethod
    def load_from_jsonl(cls, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                custom_id = entry['custom_id']
                name = custom_id.split('-')[1].replace('_', ' ')
                content = entry['response']['body']['choices'][0]['message']['content']
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {'content': content}
                cls.objects.update_or_create(
                    custom_id=custom_id,
                    defaults={'name': name, 'content': content}
                )


class GroundTruthData(models.Model):
    binomial_name = models.CharField(max_length=255, unique=True)
    ncbi_id = models.IntegerField()
    motility = models.BooleanField()
    gram_staining = models.CharField(max_length=50)
    aerophilicity = models.CharField(max_length=50, null=True, blank=True)
    extreme_environment_tolerance = models.BooleanField(null=True, blank=True)
    biofilm_formation = models.BooleanField(null=True, blank=True)
    animal_pathogenicity = models.BooleanField(null=True, blank=True)
    biosafety_level = models.CharField(max_length=50)
    health_association = models.BooleanField(null=True, blank=True)
    host_association = models.BooleanField(null=True, blank=True)
    plant_pathogenicity = models.BooleanField(null=True, blank=True)
    spore_formation = models.BooleanField(null=True, blank=True)
    hemolysis = models.BooleanField(null=True, blank=True)
    cell_shape = models.CharField(max_length=50)
    member_of_wa_subset = models.BooleanField()
    superkingdom = models.CharField(max_length=50)
    phylum = models.CharField(max_length=50)
    class_field = models.CharField(max_length=50)
    order = models.CharField(max_length=50)
    family = models.CharField(max_length=50)
    genus = models.CharField(max_length=50)
    species = models.CharField(max_length=50)
    ftp_path = models.URLField(null=True, blank=True)
    fasta_file = models.CharField(max_length=255, null=True, blank=True)


    def __str__(self):
        return self.binomial_name