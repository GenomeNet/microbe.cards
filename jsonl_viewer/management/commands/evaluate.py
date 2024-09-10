from django.core.management.base import BaseCommand
from django.db.models import Prefetch
from jsonl_viewer.models import PhenotypeDefinition, Phenotype, PredictedPhenotype, Prediction
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_score, balanced_accuracy_score
import pandas as pd
import numpy as np

def convert_to_logical(series):
    return series.map({'TRUE': True, 'FALSE': False, True: True, False: False, 'true': True, 'false': False}, na_action='ignore')

def calc_metrics(pred, true):
    pred = pd.Series(pred)
    true = pd.Series(true)
    
    valid_mask = ~(pred.isna() | true.isna())
    pred = pred[valid_mask]
    true = true[valid_mask]
    
    if len(pred) == 0 or len(set(true)) < 2:
        return {'balanced_accuracy': np.nan, 'precision': np.nan}
    
    # Check if the data is binary
    is_binary = set(pred.unique()) | set(true.unique()) <= {'TRUE', 'FALSE', True, False, 'true', 'false'}
    
    if is_binary:
        # Binary case
        pred = convert_to_logical(pred)
        true = convert_to_logical(true)
        balanced_acc = balanced_accuracy_score(true, pred)
        precision = precision_score(true, pred, average='binary', zero_division=0)
    else:
        # Multi-class case
        all_labels = sorted(set(true) | set(pred))
        label_to_int = {label: i for i, label in enumerate(all_labels)}
        
        true_encoded = true.map(label_to_int)
        pred_encoded = pred.map(label_to_int)
        
        cm = confusion_matrix(true_encoded, pred_encoded)
        
        # Calculate balanced accuracy
        balanced_acc_per_class = []
        for i in range(len(all_labels)):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - TP - FN - FP
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            balanced_acc_per_class.append((sensitivity + specificity) / 2)
        
        balanced_acc = np.mean(balanced_acc_per_class)
        
        # Use macro-averaged precision for multi-class
        precision = precision_score(true_encoded, pred_encoded, average='macro', zero_division=0)
    
    return {'balanced_accuracy': balanced_acc, 'precision': precision}

class Command(BaseCommand):
    help = 'Calculate metrics for model predictions'

    def handle(self, *args, **options):
        phenotypes = PhenotypeDefinition.objects.exclude(name="Member of WA subset")
        results = []
        dataset_sizes = []

        for phenotype in phenotypes:
            self.stdout.write(f"Calculating metrics for phenotype: {phenotype.name}")
    
            ground_truth = Phenotype.objects.filter(definition=phenotype).values('microbe__binomial_name', 'value')
            ground_truth_count = ground_truth.count()
            
            predictions = PredictedPhenotype.objects.filter(definition=phenotype).select_related('prediction')
            predictions_count = predictions.count()
            
            ground_truth_df = pd.DataFrame(ground_truth)
            ground_truth_df.columns = ['Binomial.name', 'true_value']

            predictions_df = pd.DataFrame(predictions.values('prediction__microbe__binomial_name', 'value', 'prediction__model'))
            predictions_df.columns = ['Binomial.name', 'pred_value', 'Model']

            merged_df = pd.merge(ground_truth_df, predictions_df, on='Binomial.name', how='inner')
            merged_count = len(merged_df)

            for model in merged_df['Model'].unique():
                if pd.isna(model):
                    continue
                model_data = merged_df[merged_df['Model'] == model]
                
                # Remove rows where either true_value or pred_value is NaN
                model_data = model_data.dropna(subset=['true_value', 'pred_value'])
                
                if len(model_data) == 0:
                    continue  # Skip if no valid data points

                # Collect dataset sizes for each class, stratified by model
                class_sizes = model_data['true_value'].value_counts().to_dict()
                dataset_sizes.append({
                    'Target': phenotype.name,
                    'Model': model,
                    'TotalSamples': len(model_data),
                    'ClassSizes': class_sizes
                })

                if phenotype.name == 'aerophilicity':
                    # Special handling for aerophilicity
                    pred_split = model_data['pred_value'].str.split(',\s*')
                    true_values = model_data['true_value']
                    pred_match = [not is_mutually_exclusive(pred, [true]) for pred, true in zip(pred_split, true_values)]
                    metrics = calc_metrics(pred_match, [True] * len(pred_match))
                else:
                    metrics = calc_metrics(model_data['pred_value'], model_data['true_value'])
                
                results.append({
                    'Model': model,
                    'Target': phenotype.name,
                    'BalancedAcc': metrics['balanced_accuracy'],
                    'Precision': metrics['precision'],
                    'SampleSize': len(model_data),
                })

            self.stdout.write(f"Completed calculations for {phenotype.name}")

        results_df = pd.DataFrame(results)
        print(results_df)
        
        # Save the results to a CSV file
        results_df.to_csv('metrics_results.csv', index=False)

        dataset_sizes_df = pd.DataFrame(dataset_sizes)
        dataset_sizes_df.to_csv('dataset_sizes.csv', index=False)

        self.stdout.write(self.style.SUCCESS('Metrics calculation completed'))

def is_mutually_exclusive(pred_list, true_list):
    # Implement the logic to check if predictions are mutually exclusive with ground truth
    # This should match the R implementation
    pass