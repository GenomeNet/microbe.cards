# About

microbe.cards displays predictions of on microbe.card Ground Truth Data Specification for model predictions using microbe.card Prediction Specification https://imgs.xkcd.com/comics/standards_2x.png

# Installation

The Django database containing the phenotypes is not included in this codebase. Please download it separately using:

`wget https://research.bifo.helmholtz-hzi.de/downloads/genomenet/microbecards/db.sqlite3`

Once this is done you can create a admin profile

```bash
python manage.py createsuperuser
```s

and run the server via 


```bash
python manage.py runserver
```


# Development

## File specifications

### microbe.card Ground Truth Data Specification

The microbe.card Ground Truth Data Specification defines a flexible and extensible structure for storing comprehensive information about microbial species and their phenotypes. This specification is designed to accommodate a wide range of phenotypic traits without requiring changes to the underlying schema. It includes detailed taxonomic information, allowing for alternative names to account for renaming events in microbial taxonomy. The heart of this specification is the dynamic 'phenotypes' object, which can store any number of phenotypic traits, each with its own value, data type, confidence score, and source information. This approach enables easy addition of new phenotypes as they become relevant to research or as new data becomes available. The specification also includes genome information, providing links to reference genomes for each species. By standardizing the storage of ground truth data, this specification facilitates data sharing, comparative studies, and serves as a reliable reference for evaluating predictive models in microbial phenotype research.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "entries": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "binomialName": {
            "type": "string",
            "description": "Current binomial name of the microbe"
          },
          "ncbiId": {
            "type": "integer",
            "description": "NCBI Taxonomy ID"
          },
          "taxonomy": {
            "type": "object",
            "properties": {
              "superkingdom": { "type": "string" },
              "phylum": { "type": "string" },
              "class": { "type": "string" },
              "order": { "type": "string" },
              "family": { "type": "string" },
              "genus": { "type": "string" },
              "species": { "type": "string" }
            },
            "required": ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]
          },
          "alternativeNames": {
            "type": "array",
            "items": { "type": "string" },
            "description": "List of other species names in case of renaming events"
          },
          "phenotypes": {
            "type": "object",
            "additionalProperties": {
              "type": "object",
              "properties": {
                "value": { "type": ["boolean", "string", "number", "null"] }
              },
              "required": ["value"]
            },
            "description": "Observed phenotypes for this microbe"
          },
          "genomeInfo": {
            "type": "object",
            "properties": {
              "ftpPath": { "type": "string" },
              "fastaFile": { "type": "string" }
            },
            "required": ["ftpPath", "fastaFile"]
          }
        },
        "required": ["binomialName", "ncbiId", "taxonomy", "phenotypes", "genomeInfo"]
      }
    },
    "dataSource": {
      "type": "string",
      "description": "Primary source of the ground truth data"
    },
    "lastUpdated": {
      "type": "string",
      "format": "date-time",
      "description": "Timestamp of the last update to this ground truth data"
    }
  },
  "required": ["entries", "dataSource", "lastUpdated"]
}
```

This file can be generated using the Table S1 from Münch et al

```
python manage.py makemigrations jsonl_viewer
python manage.py migrate
python manage.py import_ground_truth data/ground_truth_data.json data/phenotypes.json
```

This will create a `phenotypes.json` that can be edited to provide futher information for each phenotype such as a description. 

### microbe.card Prediction Specification

The microbe.card Prediction Specification outlines a structured format for recording and sharing predictions of microbial phenotypes. This specification is designed to work in conjunction with the Ground Truth Data Specification, allowing for easy comparison and evaluation of predictive models. The format accommodates predictions for multiple phenotypes across various microbial species in a single dataset. It includes metadata about the prediction run, such as the model name, version, and timestamp, providing important context for the predictions. Each prediction entry is linked to a specific microbial species via its binomial name and NCBI Taxonomy ID, ensuring clear association with ground truth data. The specification allows for different types of predicted values to match the variety of phenotypic traits, and includes confidence scores for each prediction. Notably, it features an option to exclude specific predictions from evaluation, offering flexibility in cases where predictions might be based on the same data as the ground truth or where there's uncertainty about the prediction. This comprehensive yet flexible structure supports a wide range of predictive modeling approaches in microbial phenotype research, facilitating model comparison, evaluation, and iterative improvement.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "predictionMetadata": {
      "type": "object",
      "properties": {
        "predictionId": {
          "type": "string",
          "description": "Unique identifier for this prediction run"
        },
        "modelName": {
          "type": "string",
          "description": "Name of the prediction model used"
        },
        "modelVersion": {
          "type": "string",
          "description": "Version of the prediction model"
        },
        "predictedPhenotypes": {
          "type": "array",
          "items": { "type": "string" },
          "description": "List of phenotypes predicted in this run"
        },
        "timestamp": {
          "type": "string",
          "format": "date-time",
          "description": "Timestamp of when this prediction run was performed"
        },
        "description": {
          "type": "string",
          "description": "Optional description of the prediction run or model"
        }
      },
      "required": ["predictionId", "modelName", "modelVersion", "predictedPhenotypes", "timestamp"]
    },
    "predictions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "binomialName": {
            "type": "string",
            "description": "Binomial name of the microbe, should match a ground truth entry"
          },
          "ncbiId": {
            "type": "integer",
            "description": "NCBI Taxonomy ID, should match a ground truth entry"
          },
          "phenotypePredictions": {
            "type": "object",
            "additionalProperties": {
              "type": "object",
              "properties": {
                "predictedValue": { 
                  "type": ["boolean", "string", "number", "null"],
                  "description": "The predicted value for the phenotype"
                },
                "confidence": { 
                  "type": "number", 
                  "minimum": 0, 
                  "maximum": 1,
                  "description": "Confidence score for the prediction"
                },
                "excludeFromEvaluation": {
                  "type": "boolean",
                  "default": false,
                  "description": "If true, this prediction should be omitted from evaluation"
                }
              },
              "required": ["predictedValue", "confidence"]
            }
          }
        },
        "required": ["binomialName", "ncbiId", "phenotypePredictions"]
      }
    }
  },
  "required": ["predictionMetadata", "predictions"]
}
```