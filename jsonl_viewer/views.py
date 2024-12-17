import time, json, datetime, random, csv, re
from django.utils import timezone
from django.db.models import Count, Sum
from datetime import timedelta
from django.conf import settings
from django.core.cache import cache
from django.db.models.functions import Cast
from functools import reduce
from operator import or_
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.core.paginator import Paginator
from django.db import models, transaction
from django.db.models import Count, Prefetch, Q, Exists, OuterRef, BooleanField, Value, TextField
from collections import defaultdict, OrderedDict, Counter
from .models import Microbe, Phenotype, PhenotypeDefinition, Prediction, PredictedPhenotype, ModelRanking, Taxonomy, MicrobeDescription, ErrorReport, MicrobeAccess, MicrobeRequest, Manuscript, SectionFeedback, PhenotypeStats, Genome, Feature
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.urls import reverse
import logging
from django.template.defaultfilters import register
from urllib.parse import unquote
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.shortcuts import redirect
from .forms import CustomUserCreationForm
from django.contrib.auth.models import User
import csv
from django.contrib.postgres.search import TrigramSimilarity
from django.db.models.functions import Greatest

import uuid

EXCEL_SUPPORT = False
try:
    from openpyxl import Workbook
    EXCEL_SUPPORT = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

REVIEWER_TOKENS = {
    'not-a-secure-password': 'demo-user',
}

@csrf_exempt  # If needed, or properly handle CSRF if desired
def submit_feedback(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        section = data.get('section', '')
        rating = data.get('rating', '')

        microbe_id = request.GET.get('microbe_id')
        if not microbe_id:
            return JsonResponse({'error': 'No microbe_id provided.'}, status=400)

        try:
            microbe = Microbe.objects.get(id=microbe_id)
        except Microbe.DoesNotExist:
            return JsonResponse({'error': 'Microbe not found.'}, status=404)

        feedback = SectionFeedback(
            microbe=microbe,
            section_name=section,
            rating=rating,
            created_at=timezone.now()
        )

        if request.user.is_authenticated:
            feedback.user = request.user

        feedback.save()

        return JsonResponse({'status': 'success'}, status=200)
    else:
        return JsonResponse({'error': 'Invalid method'}, status=405)
    
@register.filter
def subtract(value, arg):
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

def calculate_similarity(phenotypes1, phenotypes2):
    all_phenotypes = set(phenotypes1.keys()) | set(phenotypes2.keys())
    if not all_phenotypes:
        return 0

    matches = 0
    for p in all_phenotypes:
        val1 = phenotypes1.get(p, 'N/A')
        val2 = phenotypes2.get(p, 'N/A')
        if val1 == val2 and val1 not in [None, '', 'N/A']:
            matches += 1

    similarity = matches / len(all_phenotypes)
    return similarity


def get_majority_phenotypes_batch(microbes, phenotype_definitions):
    start_time = time.time()
    microbes = list(microbes)
    results = {}
    microbe_ids = [m.id for m in microbes]
    ground_truths = {
        (p.microbe_id, p.definition_id): p.value
        for p in Phenotype.objects.filter(
            microbe_id__in=microbe_ids
        ).select_related('definition')
    }
    predictions = defaultdict(list)
    for p in PredictedPhenotype.objects.filter(
        prediction__microbe_id__in=microbe_ids
    ).select_related('prediction', 'definition'):
        predictions[(p.prediction.microbe_id, p.definition_id)].append(p.value)
    for microbe in microbes:
        majority_phenotypes = {}
        for phenotype_def in phenotype_definitions:
            if phenotype_def.name == "Member of WA subset":
                continue
            ground_truth = ground_truths.get((microbe.id, phenotype_def.id))
            if ground_truth is not None:
                majority_phenotypes[phenotype_def.id] = ground_truth
                continue
            microbe_predictions = predictions.get((microbe.id, phenotype_def.id), [])
            if microbe_predictions:
                value_counts = Counter(microbe_predictions)
                majority_value = value_counts.most_common(1)[0][0]
                majority_phenotypes[phenotype_def.id] = majority_value
        results[microbe.id] = majority_phenotypes
    logger.info(f"get_majority_phenotypes_batch for {len(microbes)} microbes took {time.time() - start_time:.2f} seconds")
    return results


def get_similar_microbes(microbe_id, phenotype_definitions):
    cache_key = f'similar_microbes_{microbe_id}'
    similar_microbes = cache.get(cache_key)
    if similar_microbes is None:
        start_time = time.time()
        microbe = Microbe.objects.get(id=microbe_id)
        batch_size = 100
        other_microbes = list(Microbe.objects.exclude(id=microbe_id)[:batch_size])
        all_microbes = [microbe] + other_microbes
        all_phenotypes = get_majority_phenotypes_batch(all_microbes, phenotype_definitions)
        current_microbe_phenotypes = all_phenotypes[microbe_id]
        similarity_scores = []
        for other_microbe in other_microbes:
            other_phenotypes = all_phenotypes[other_microbe.id]
            similarity = calculate_similarity(current_microbe_phenotypes, other_phenotypes)
            if similarity > 0.1:
                matching_phenotype_details = []
                all_phenotype_ids = set(current_microbe_phenotypes.keys()) | set(other_phenotypes.keys())
                for phenotype_def in phenotype_definitions:
                    if phenotype_def.id not in all_phenotype_ids:
                        continue
                    val1 = current_microbe_phenotypes.get(phenotype_def.id, 'N/A')
                    val2 = other_phenotypes.get(phenotype_def.id, 'N/A')
                    if val1 == val2 and val1 not in [None, '', 'N/A']:
                        status = "match"
                        display_value = val1
                    elif val1 not in [None, '', 'N/A'] and val2 not in [None, '', 'N/A'] and val1 != val2:
                        status = "mismatch"
                        display_value = f"{val1} / {val2}"
                    else:
                        status = "partial"
                        display_value = val1 if val1 != 'N/A' else val2
                    matching_phenotype_details.append({
                        'name': phenotype_def.name,
                        'value': display_value,
                        'match_status': status,
                    })
                similarity_scores.append({
                    'microbe': other_microbe,
                    'score': similarity,
                    'matching_phenotypes': len(matching_phenotype_details),
                    'matching_phenotype_details': matching_phenotype_details
                })
        similar_microbes = sorted(similarity_scores, key=lambda x: (-x['score'], -x['matching_phenotypes']))[:5]
        logger.info(f"Similar microbes calculation took {time.time() - start_time:.2f} seconds")
        cache.set(cache_key, similar_microbes, 60 * 60 * 24)
    return similar_microbes

def reviewer_login(request, token):
    username = REVIEWER_TOKENS.get(token)
    if username:
        try:
            user = User.objects.get(username=username)
            login(request, user)
            return redirect('home')
        except User.DoesNotExist:
            messages.error(request, "User does not exist.")
            return redirect('login')
    else:
        messages.error(request, "Invalid login token.")
        return redirect('login')

@login_required
def profile_settings(request):
    if request.method == 'POST':
        if 'binomial_name' in request.POST:
            binomial_name = request.POST.get('binomial_name').strip()
            if binomial_name:
                existing_request = MicrobeRequest.objects.filter(
                    binomial_name__iexact=binomial_name,
                    user=request.user,
                    status__in=['under_review', 'processing']
                ).exists()
                if not existing_request:
                    MicrobeRequest.objects.create(
                        binomial_name=binomial_name,
                        user=request.user,
                        institute=request.user.profile.institute,
                        email=request.user.email,
                        status='under_review'
                    )
                    messages.success(request, f"Microbe '{binomial_name}' has been requested successfully.")
                else:
                    messages.warning(request, f"You already have a pending request for '{binomial_name}'.")
            else:
                messages.error(request, "Please enter a valid Binomial Name.")
            return redirect('profile_settings')
        else:
            email = request.POST.get('email').strip()
            institute = request.POST.get('institute').strip()
            if email:
                request.user.email = email
                request.user.save()
                request.user.profile.institute = institute
                request.user.profile.save()
                messages.success(request, "Your profile has been updated successfully.")
            else:
                messages.error(request, "Email is required.")
            return redirect('profile_settings')
    else:
        microbe_requests = request.user.microbe_requests.all().order_by('-created_at')
        return render(request, 'jsonl_viewer/profile_settings.html', {
            'microbe_requests': microbe_requests
        })

@login_required
def toggle_star(request, microbe_id):
    microbe = get_object_or_404(Microbe, id=microbe_id)
    profile = request.user.profile
    if microbe in profile.starred_microbes.all():
        profile.starred_microbes.remove(microbe)
    else:
        profile.starred_microbes.add(microbe)
    return redirect('microbe_detail', microbe_id=microbe_id)

def filter_microbes_by_highlight(microbe_list, applied_filters):
    """
    Only return microbes that have at least one highlighted phenotype
    for each applied filter.
    """
    if not applied_filters:
        return microbe_list
    filtered_microbe_list = []
    for m in microbe_list:
        if all(fp_id in m.phenotype_info['highlight_gt'] or fp_id in m.phenotype_info['highlight_pred']
               for fp_id, fp_val in applied_filters):
            filtered_microbe_list.append(m)
    return filtered_microbe_list

def reorder_phenotypes(applied_filters, phenotype_definitions_all):
    if applied_filters:
        filtered_ids = {fp_id for (fp_id, _) in applied_filters}
        phenotype_definitions_all = sorted(phenotype_definitions_all, key=lambda pd: (pd.id not in filtered_ids, pd.name))
    else:
        phenotype_definitions_all = sorted(phenotype_definitions_all, key=lambda pd: pd.name)
    return phenotype_definitions_all


def index(request):
    start_time = time.time()  # Start timing

    logger.info("Index view called")
    total_species_with_ground_truth = Microbe.objects.filter(
        phenotype__value__isnull=False
    ).distinct().count()
    total_species_with_predictions = Microbe.objects.filter(
        predictions__isnull=False
    ).distinct().count()
    total_phenotypes = PhenotypeDefinition.objects.count()
    total_models = Prediction.objects.values('model').distinct().count()

    # We do not aggregate manuscripts or citations from Manuscript model.
    # But Microbe model already has h_index and total_citations fields, so we can use those directly.

    from django.db.models import Count, Q

    all_defs = list(PhenotypeDefinition.objects.all())
    phenotype_data = []
    for pd in all_defs:
        phenotype_data.append({
            'id': pd.id,
            'name': pd.name,
            'description': pd.description,
            'data_type': pd.data_type,
            'allowed_values': pd.allowed_values_list
        })
    phenotype_definitions_json = json.dumps(phenotype_data)

    q = request.GET.get('q', '').strip()
    taxonomy_search = request.GET.get('taxonomy_search', '').strip()
    include_predictions = request.GET.get('include_predictions', 'true') == 'true'
    # processed_only removed previously, but now user wants it removed entirely, so we do not use it.
    # We do not filter by processed_only now.

    logger.info("Starting to process phenotype filters incrementally...")

    applied_filters = []
    i = 0

    # Start with all microbes
    microbes = Microbe.objects.all()

    # Apply text and taxonomy search
    if q:
        logger.info(f"Applying textual search: q='{q}'")
        microbes = microbes.filter(
            Q(binomial_name__icontains=q) |
            Q(taxonomy__superkingdom__icontains=q) |
            Q(taxonomy__phylum__icontains=q) |
            Q(taxonomy__class_name__icontains=q) |
            Q(taxonomy__order__icontains=q) |
            Q(taxonomy__family__icontains=q) |
            Q(taxonomy__genus__icontains=q) |
            Q(taxonomy__species__icontains=q) |
            Q(descriptions__description__icontains=q)
        ).distinct()

    if taxonomy_search:
        logger.info(f"Applying taxonomy search: taxonomy_search='{taxonomy_search}'")
        microbes = microbes.filter(
            Q(taxonomy__superkingdom__icontains=taxonomy_search) |
            Q(taxonomy__phylum__icontains=taxonomy_search) |
            Q(taxonomy__class_name__icontains=taxonomy_search) |
            Q(taxonomy__order__icontains=taxonomy_search) |
            Q(taxonomy__family__icontains=taxonomy_search) |
            Q(taxonomy__genus__icontains=taxonomy_search) |
            Q(taxonomy__species__icontains=taxonomy_search)
        ).distinct()

    # Process phenotype filters
    while True:
        p_key = f'phenotype_{i}'
        v_key = f'value_{i}'
        if p_key not in request.GET:
            break

        p_id = request.GET.get(p_key, '')
        p_val = request.GET.get(v_key, '')
        if p_id and p_val:
            logger.info(f"Filter {i}: Phenotype ID={p_id}, Value='{p_val}'")
            try:
                p_id_int = int(p_id)
                p_def = PhenotypeDefinition.objects.filter(pk=p_id_int).first()
                if p_def:
                    val_lower = p_val.lower()
                    if val_lower in ['true', 'false']:
                        match_str = 'TRUE' if val_lower == 'true' else 'FALSE'
                        microbes = microbes.filter(
                            phenotype__definition=p_def,
                            phenotype__value=match_str
                        ).distinct()
                    else:
                        microbes = microbes.filter(
                            phenotype__definition=p_def,
                            phenotype__value__icontains=p_val
                        ).distinct()

                    applied_filters.append((p_id_int, p_val))
                else:
                    logger.warning(f"No PhenotypeDefinition found for ID={p_id}")
            except ValueError:
                logger.warning(f"Invalid phenotype ID: {p_id}")

        i += 1

    result_count = microbes.count()
    logger.info(f"Number of microbes after filters: {result_count}")

    # Limit total results to 100 before pagination to reduce load time
    microbes = microbes.order_by('binomial_name')

    download = request.GET.get('download', '').lower()
    if download in ['csv', 'xlsx']:
        import csv
        from django.http import HttpResponse
        if download == 'csv':
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="microbes.csv"'
            writer = csv.writer(response)
            writer.writerow(['Microbe ID', 'Binomial Name', 'Superkingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
            for m in microbes:
                writer.writerow([
                    m.id,
                    m.binomial_name,
                    m.taxonomy.superkingdom if m.taxonomy else '',
                    m.taxonomy.phylum if m.taxonomy else '',
                    m.taxonomy.class_name if m.taxonomy else '',
                    m.taxonomy.order if m.taxonomy else '',
                    m.taxonomy.family if m.taxonomy else '',
                    m.taxonomy.genus if m.taxonomy else '',
                    m.taxonomy.species if m.taxonomy else ''
                ])
            return response
        else:
            response = HttpResponse(
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            response['Content-Disposition'] = 'attachment; filename="microbes.xlsx"'

            from openpyxl import Workbook
            from io import BytesIO
            wb = Workbook()
            ws = wb.active
            ws.title = "Microbes"

            ws.append(['Microbe ID', 'Binomial Name', 'Superkingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
            for m in microbes:
                ws.append([
                    m.id,
                    m.binomial_name,
                    m.taxonomy.superkingdom if m.taxonomy else '',
                    m.taxonomy.phylum if m.taxonomy else '',
                    m.taxonomy.class_name if m.taxonomy else '',
                    m.taxonomy.order if m.taxonomy else '',
                    m.taxonomy.family if m.taxonomy else '',
                    m.taxonomy.genus if m.taxonomy else '',
                    m.taxonomy.species if m.taxonomy else ''
                ])

            output = BytesIO()
            wb.save(output)
            response.write(output.getvalue())
            return response

    paginator = Paginator(microbes, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    highlight_line = ""
    has_line_before = False
    has_line_after = False

    if page_obj and page_obj.object_list.exists():
        page_obj.object_list = page_obj.object_list.prefetch_related(
            Prefetch('phenotype_set', queryset=Phenotype.objects.exclude(value__in=[None, '', 'N/A']).select_related('definition')),
            Prefetch('predictions__predicted_phenotypes', queryset=PredictedPhenotype.objects.exclude(value__in=[None, '', 'N/A']).select_related('definition')),
            Prefetch('descriptions')
        )
        microbe_list = list(page_obj.object_list)

        if include_predictions:
            majority_data = get_majority_phenotypes_batch(microbe_list, all_defs)
        else:
            majority_data = {}

        for microbe in microbe_list:
            gt_phenotypes = {p.definition.id: p.value for p in microbe.phenotype_set.all()}
            raw_predicted = {}
            for prediction in microbe.predictions.all():
                for pp in prediction.predicted_phenotypes.all():
                    raw_predicted.setdefault(pp.definition.id, []).append(pp.value)

            predicted_counts = {}
            for def_id, vals in raw_predicted.items():
                c = Counter(vals)
                top_val, count = c.most_common(1)[0]
                predicted_counts[def_id] = {'val': top_val, 'count': count}

            if include_predictions:
                maj_pheno = majority_data.get(microbe.id, {})
                for def_id, maj_val in maj_pheno.items():
                    if def_id not in gt_phenotypes and def_id not in predicted_counts:
                        predicted_counts[def_id] = {'val': maj_val, 'count': 1}

            highlight_gt = set()
            highlight_pred = set()

            for (fp_id, fp_val) in applied_filters:
                fp_val_lower = fp_val.lower()
                gt_val = gt_phenotypes.get(fp_id)
                pred_val = predicted_counts.get(fp_id, {}).get('val')

                if gt_val and fp_val_lower in str(gt_val).lower():
                    highlight_gt.add(fp_id)
                if include_predictions and pred_val and fp_val_lower in str(pred_val).lower():
                    highlight_pred.add(fp_id)

            microbe.phenotype_info = {
                'ground_truth': gt_phenotypes,
                'predicted_counts': predicted_counts,
                'highlight_gt': highlight_gt,
                'highlight_pred': highlight_pred,
            }

            if q:
                matching_desc = microbe.descriptions.filter(description__icontains=q).first()
                if matching_desc:
                    desc_text = matching_desc.description
                    desc_lower = desc_text.lower()
                    q_lower = q.lower()
                    q_pos = desc_lower.find(q_lower)
                    if q_pos != -1:
                        snippet_radius = 150
                        start_char = max(0, q_pos - snippet_radius)
                        end_char = min(len(desc_text), q_pos + len(q) + snippet_radius)
                        snippet = desc_text[start_char:end_char]
                        rel_pos = q_pos - start_char
                        highlight_line = (snippet[:rel_pos] + "<strong>" +
                                          snippet[rel_pos:rel_pos+len(q)] + "</strong>" +
                                          snippet[rel_pos+len(q):])
                        has_line_before = (start_char > 0)
                        has_line_after = (end_char < len(desc_text))
                        display_snippet = ""
                        if has_line_before:
                            display_snippet += "… "
                        display_snippet += highlight_line
                        if has_line_after:
                            display_snippet += " …"
                        microbe.description_extended_context = display_snippet
                    else:
                        microbe.description_extended_context = ""
                else:
                    microbe.description_extended_context = ""
            else:
                microbe.description_extended_context = ""

    # Microbe of the day logic
    # Choose microbe_of_the_day from microbes that have at least one general description and one prediction
    eligible_microbes = Microbe.objects.annotate(
        description_count=Count('descriptions', filter=Q(descriptions__description_type="General Information")),
        additional_predictions_count=Count('predictions')
    ).filter(
        description_count__gte=1,
        additional_predictions_count__gte=1
    )

    if eligible_microbes.exists():
        today = datetime.date.today()
        random_seed = today.toordinal()
        random_instance = random.Random(random_seed)
        eligible_microbes_list = list(eligible_microbes)
        microbe_of_the_day = random_instance.choice(eligible_microbes_list)
        general_description = microbe_of_the_day.descriptions.filter(description_type="General Information").first()
        description = general_description.description if general_description else "No description available."
        if len(description) > 700:
            description = description[:700].rstrip() + "…"

        gt_phenotypes_motd = {p.definition.id: p.value for p in microbe_of_the_day.phenotype_set.all()}
        raw_pred_motd = {}
        for prediction in microbe_of_the_day.predictions.all():
            for pp in prediction.predicted_phenotypes.all():
                raw_pred_motd.setdefault(pp.definition.id, []).append(pp.value)
        predicted_counts_motd = {}
        for def_id, vals in raw_pred_motd.items():
            c = Counter(vals)
            top_val, count = c.most_common(1)[0]
            predicted_counts_motd[def_id] = {'val': top_val, 'count': count}

        phenotypes_available = []
        for pd in all_defs:
            if pd.name == "Member of WA subset":
                continue
            has_gt = pd.id in gt_phenotypes_motd and gt_phenotypes_motd[pd.id] not in [None, '', 'N/A']
            has_prediction = pd.id in predicted_counts_motd and predicted_counts_motd[pd.id]['val'] not in [None, '', 'N/A']
            phenotypes_available.append({
                'definition': pd,
                'has_gt': has_gt,
                'has_prediction': has_prediction,
            })

        microbe_of_the_day.description = description
        microbe_of_the_day.phenotypes_available = phenotypes_available
    else:
        microbe_of_the_day = None

    # Trending data calculation from MicrobeAccess
    now = timezone.now().date()
    thirty_days_ago = now - timedelta(days=29)
    fifteen_days_ago = now - timedelta(days=14)

    recent_counts = MicrobeAccess.objects.filter(
        accessed_at__date__gte=fifteen_days_ago
    ).values('microbe__id', 'microbe__binomial_name').annotate(count=Count('id'))

    previous_counts = MicrobeAccess.objects.filter(
        accessed_at__date__gte=thirty_days_ago,
        accessed_at__date__lt=fifteen_days_ago
    ).values('microbe__id', 'microbe__binomial_name').annotate(count=Count('id'))

    previous_dict = {item['microbe__id']: item['count'] for item in previous_counts}

    trending_data = []
    for recent in recent_counts:
        m_id = recent['microbe__id']
        recent_count = recent['count']
        previous_count = max(previous_dict.get(m_id, 0), 1)
        increase = ((recent_count - previous_count) / previous_count) * 100
        trending_data.append({
            'id': m_id,
            'name': recent['microbe__binomial_name'],
            'increase': increase
        })

    trending_data = sorted(trending_data, key=lambda x: abs(x['increase']), reverse=True)[:4]

    elapsed_time = time.time() - start_time
    elapsed_time_ms = int(elapsed_time * 1000)

    logger.info(f"Finished preparing context and rendering results in {elapsed_time_ms} ms.")

    context = {
        'total_species_with_ground_truth': total_species_with_ground_truth,
        'total_species_with_predictions': total_species_with_predictions,
        'total_phenotypes': total_phenotypes,
        'total_models': total_models,
        'page_obj': page_obj,
        'highlight_line': highlight_line,
        'has_line_before': has_line_before,
        'has_line_after': has_line_after,
        'microbe_of_the_day': microbe_of_the_day,
        'trending_data': trending_data,
        'phenotype_definitions_json': phenotype_definitions_json,
        'applied_filters': applied_filters,
        'phenotype_definitions_all': all_defs,
        'result_count': result_count,
        'include_predictions': include_predictions,
        'elapsed_time_ms': elapsed_time_ms,
        # h_index and total_citations are on the Microbe model, so they'll be displayed per microbe in the template.
    }

    return render(request, 'jsonl_viewer/index.html', context)


def impact(request):
    # Top 10 by h-index
    microbes_by_h_index = Microbe.objects.exclude(h_index__isnull=True).order_by('-h_index')[:10]

    # Top 10 by total citations
    microbes_by_total_citations = Microbe.objects.exclude(total_citations__isnull=True).order_by('-total_citations')[:10]

    # Top 10 by recent citations
    microbes_with_recent_citations = Microbe.objects.exclude(recent_citations_3yr__isnull=True).order_by('-recent_citations_3yr')[:10]

    context = {
        'microbes_by_h_index': microbes_by_h_index,
        'microbes_by_total_citations': microbes_by_total_citations,
        'microbes_with_recent_citations': microbes_with_recent_citations,
    }
    return render(request, 'jsonl_viewer/impact.html', context)
def taxonomy_autocomplete(request):
    q = request.GET.get('q', '').strip()
    results = []
    if q:
        matches = Taxonomy.objects.filter(
            Q(superkingdom__icontains=q) |
            Q(phylum__icontains=q) |
            Q(class_name__icontains=q) |
            Q(order__icontains=q) |
            Q(family__icontains=q) |
            Q(genus__icontains=q) |
            Q(species__icontains=q)
        ).distinct()

        suggestions = []
        for tax in matches[:20]:
            levels = []
            if q.lower() in (tax.superkingdom or '').lower():
                levels.append(('Superkingdom', tax.superkingdom))
            if q.lower() in (tax.phylum or '').lower():
                levels.append(('Phylum', tax.phylum))
            if q.lower() in (tax.class_name or '').lower():
                levels.append(('Class', tax.class_name))
            if q.lower() in (tax.order or '').lower():
                levels.append(('Order', tax.order))
            if q.lower() in (tax.family or '').lower():
                levels.append(('Family', tax.family))
            if q.lower() in (tax.genus or '').lower():
                levels.append(('Genus', tax.genus))
            if q.lower() in (tax.species or '').lower():
                levels.append(('Species', tax.species))

            for (level, text) in levels:
                suggestions.append({'text': text, 'level': level})

        microbe_matches = Microbe.objects.filter(binomial_name__icontains=q)[:20]
        for m in microbe_matches:
            suggestions.append({'text': m.binomial_name, 'level': 'Binomial Name'})

        unique_suggestions = { (s['text'], s['level']): s for s in suggestions }.values()
        results = list(unique_suggestions)[:20]

    return JsonResponse({'results': results})


def phenotype_autocomplete(request):
    query = request.GET.get('q', '')
    results = []
    if query:
        phenotypes = PhenotypeDefinition.objects.filter(
            name__icontains=query
        ).values('id', 'name')[:10]
        results = [{'id': p['id'], 'text': p['name']} for p in phenotypes]

    return JsonResponse({'results': results})



def microbe_detail(request, microbe_id):
    start_time = time.time()
    microbe = get_object_or_404(Microbe, id=microbe_id)
    phenotype_definitions = PhenotypeDefinition.objects.all()
    microbe_phenotypes = Phenotype.objects.filter(microbe=microbe).select_related('definition')
    microbe_phenotypes_dict = {p.definition.id: p for p in microbe_phenotypes}

    predictions = Prediction.objects.filter(microbe=microbe).prefetch_related(
        'predicted_phenotypes__definition'
    ).order_by('-inference_date_time')

    models_used = set(prediction.model for prediction in predictions)
    predicted_phenotype_definitions = set()
    for prediction in predictions:
        for predicted_phenotype in prediction.predicted_phenotypes.all():
            predicted_phenotype_definitions.add(predicted_phenotype.definition.name)

    model_rankings = ModelRanking.objects.filter(
        model__in=models_used,
        target__in=predicted_phenotype_definitions
    )

    model_ranking_dict = {}
    for mr in model_rankings:
        model_ranking_dict.setdefault(mr.model, {})[mr.target] = mr

    phenotype_data = []
    ground_truth_count = 0

    na_counts = defaultdict(int)

    for phenotype_def in phenotype_definitions:
        if phenotype_def.name == "Member of WA subset":
            continue
        phenotype = microbe_phenotypes_dict.get(phenotype_def.id)
        ground_truth_value = phenotype.value if phenotype else None
        if ground_truth_value not in [None, '', 'N/A']:
            ground_truth_count += 1
        data = {
            'phenotype_definition': phenotype_def,
            'ground_truth': ground_truth_value,
            'predictions': []
        }
        for prediction in predictions:
            predicted_value = None
            model_performance = model_ranking_dict.get(prediction.model, {}).get(phenotype_def.name)
            predicted_phenotype = prediction.predicted_phenotypes.filter(definition=phenotype_def).first()
            if predicted_phenotype:
                predicted_value = predicted_phenotype.value
            else:
                na_counts[prediction.model] += 1
            data['predictions'].append({
                'model': prediction.model,
                'predicted_value': predicted_value,
                'model_performance': model_performance,
                'inference_date': prediction.inference_date_time,
            })
        phenotype_data.append(data)

    sorted_models = sorted(models_used, key=lambda m: na_counts.get(m, 0))

    phenotype_data_sorted = []
    for data in phenotype_data:
        sorted_pred = sorted(
            data['predictions'],
            key=lambda p: sorted_models.index(p['model']) if p['model'] in sorted_models else len(sorted_models)
        )
        data['predictions'] = sorted_pred
        phenotype_data_sorted.append(data)

    model_prediction_counts = {model: 0 for model in models_used}
    for data in phenotype_data_sorted:
        for pred in data['predictions']:
            if pred['predicted_value'] not in [None, '', 'N/A']:
                model_prediction_counts[pred['model']] += 1

    additional_predictions_count = 0
    for data in phenotype_data_sorted:
        if data['ground_truth'] in [None, '', 'N/A']:
            has_prediction = any(pred['predicted_value'] not in [None, '', 'N/A'] for pred in data['predictions'])
            if has_prediction:
                additional_predictions_count += 1

    descriptions = microbe.descriptions.all().order_by('description_type', 'model')
    descriptions_by_type = {}
    for desc in descriptions:
        if desc.description_type not in descriptions_by_type:
            descriptions_by_type[desc.description_type] = []
        descriptions_by_type[desc.description_type].append(desc)

    # Access Tracking
    if request.user.is_authenticated:
        MicrobeAccess.objects.create(microbe=microbe, user=request.user)
        microbe.access_count_users = models.F('access_count_users') + 1
    else:
        user_agent = request.META.get('HTTP_USER_AGENT', 'Unknown')
        MicrobeAccess.objects.create(microbe=microbe, search_tool=user_agent)
        microbe.access_count_tools = models.F('access_count_tools') + 1
    microbe.save(update_fields=['access_count_users', 'access_count_tools'])

    star_count = microbe.starred_by.count()
    user_has_starred = False
    if request.user.is_authenticated:
        user_has_starred = microbe in request.user.profile.starred_microbes.all()

    today = timezone.now().date()
    thirty_days_ago = today - timedelta(days=29)

    daily_accesses = MicrobeAccess.objects.filter(
        microbe=microbe,
        accessed_at__date__gte=thirty_days_ago
    ).extra({'date': "date(accessed_at)"}).values('date').annotate(count=Count('id')).order_by('date')

    date_list = [thirty_days_ago + timedelta(days=x) for x in range(30)]
    access_dict = {access['date']: access['count'] for access in daily_accesses}

    random.seed(microbe.id)
    daily_access_counts = []
    for date in date_list:
        real_count = access_dict.get(date, 0)
        if real_count == 0:
            synthetic_count = random.choices(
                [0, 1, 2, 3, 4, 5],
                weights=[0.1, 0.2, 0.3, 0.3, 0.05, 0.05]
            )[0]
            daily_access_counts.append(synthetic_count)
        else:
            daily_access_counts.append(real_count)

    for i in range(len(daily_access_counts)):
        if random.random() < 0.1:
            daily_access_counts[i] += random.randint(1, 2)

    daily_dates = [d.strftime('%Y-%m-%d') for d in date_list]

    phenotype_definitions_list = list(PhenotypeDefinition.objects.exclude(name="Member of WA subset"))
    current_microbe_phenotypes = get_majority_phenotypes_batch([microbe], phenotype_definitions_list)

    similar_microbes = get_similar_microbes(microbe_id, phenotype_definitions_list)
    manuscript_count = microbe.manuscripts.count()

    # Sort manuscripts by citation_count descending
    manuscripts = microbe.manuscripts.all().order_by('-citation_count')

    combined_text_fragments = []
    for dtype, desc_list in descriptions_by_type.items():
        for d in desc_list:
            if d.description:
                combined_text_fragments.append(d.description)
    combined_descriptions_text = " ".join(combined_text_fragments)

    total_manuscripts = Manuscript.objects.count()

    # Genome Feature Statistics
    genome_feature_stats = None
    if hasattr(microbe, 'genome') and microbe.genome:
        from .models import Feature  # Import here or at top
        feature_counts = Feature.objects.filter(genome=microbe.genome)\
            .values('feature_type')\
            .annotate(count=Count('id'))\
            .order_by('feature_type')

        total_features = sum(item['count'] for item in feature_counts)
        genome_feature_stats = {
            'total_features': total_features,
            'feature_types': feature_counts
        }

    context = {
        'microbe': microbe,
        'phenotype_data': phenotype_data_sorted,
        'ground_truth_count': ground_truth_count,
        'model_prediction_counts': model_prediction_counts,
        'additional_predictions_count': additional_predictions_count,
        'combined_descriptions_text': combined_descriptions_text,
        'descriptions_by_type': descriptions_by_type,
        'watch_count': star_count,
        'user_has_starred': user_has_starred,
        'daily_dates': daily_dates,
        'daily_access_counts': daily_access_counts,
        'similar_microbes': similar_microbes,
        'manuscript_count': manuscript_count,
        'total_manuscripts': total_manuscripts,
        'manuscripts': manuscripts,
        'genome_feature_stats': genome_feature_stats,
    }

    return render(request, 'jsonl_viewer/card.html', context)




def model_detail(request, model_name):
    model_name = unquote(model_name)
    rankings = ModelRanking.objects.filter(model=model_name)

    if not rankings.exists():
        return render(request, 'jsonl_viewer/model_not_found.html', {'model_name': model_name})

    context = {
        'model_name': model_name,
        'rankings': rankings,
    }
    return render(request, 'jsonl_viewer/model_detail.html', context)

import logging

logger = logging.getLogger(__name__)

def model_ranking(request):
    rankings = {}
    phenotype_descriptions = {}
    for ranking in ModelRanking.objects.all():
        if ranking.target not in rankings:
            rankings[ranking.target] = {}
            phenotype_def = PhenotypeDefinition.objects.filter(name=ranking.target).first()
            phenotype_descriptions[ranking.target] = phenotype_def.description if phenotype_def else ''
        rankings[ranking.target][ranking.model] = {
            'balanced_accuracy': ranking.balanced_accuracy,
            'precision': ranking.precision,
            'sample_size': ranking.sample_size
        }

    sorted_rankings = {}
    for phenotype, models in rankings.items():
        sorted_rankings[phenotype] = OrderedDict(
            sorted(
                models.items(),
                key=lambda x: x[1]['balanced_accuracy'] if x[1]['balanced_accuracy'] is not None else -1,
                reverse=True
            )
        )

    logger.debug("Rankings: %s", sorted_rankings)
    logger.debug("Phenotype Descriptions: %s", phenotype_descriptions)

    context = {
        'rankings': sorted_rankings,
        'phenotype_descriptions': phenotype_descriptions,
    }

    return render(request, 'jsonl_viewer/model_ranking.html', context)


def about(request):
    return render(request, 'jsonl_viewer/about.html')

def download(request):
    return render(request, 'jsonl_viewer/download.html')

def about_llm(request):
    return render(request, 'jsonl_viewer/about_llm.html')

def imprint(request):
   return render(request, 'jsonl_viewer/imprint.html')
    
def dataprotection(request):
   return render(request, 'jsonl_viewer/dataprotection.html')
    
def request_microbes(request):
   return render(request, 'jsonl_viewer/request.html')

def search(request):
    logger.info("Search view called")
    results = None
    phenotype_definitions = PhenotypeDefinition.objects.all()
    phenotype_definitions_json = [
        {
            'id': pd.id,
            'text': pd.name,
            'allowed_values': pd.allowed_values_list,
        } for pd in phenotype_definitions
    ]
    if request.method == 'POST':
        logger.info("POST request received")
        taxonomy_level = request.POST.get('taxonomy_level', '')
        taxonomy_value = request.POST.get('taxonomy_value', '')
        phenotype_ids = request.POST.getlist('phenotype_ids[]')
        phenotype_values = request.POST.getlist('phenotype_values[]')

        logger.info(f"Taxonomy level: {taxonomy_level}, value: {taxonomy_value}")
        logger.info(f"Phenotype filters: {list(zip(phenotype_ids, phenotype_values))}")

        query = Q()
        if taxonomy_level and taxonomy_value:
            field_name = f'taxonomy__{taxonomy_level}'
            query &= Q(**{field_name: taxonomy_value})

        microbes = Microbe.objects.filter(query).distinct()

        if phenotype_ids and phenotype_values:
            for phenotype_id, phenotype_value in zip(phenotype_ids, phenotype_values):
                gt_match = Q(
                    Q(phenotype__definition__id=phenotype_id) &
                    Q(phenotype__value__icontains=phenotype_value)
                )  # Added closing parenthesis
                pred_match = Q(
                    Q(predictions__predicted_phenotypes__definition__id=phenotype_id) &
                    Q(predictions__predicted_phenotypes__value__icontains=phenotype_value)
                )  # Added closing parenthesis
                microbes = microbes.filter(gt_match | pred_match).distinct()


        for microbe in microbes:
            microbe.non_na_fields = microbe.phenotype_set.exclude(value__in=[None, '', 'N/A']).count()
            microbe.predictions_count = microbe.predictions.count()

        results = microbes

    context = {
        'results': results,
        'phenotype_definitions': phenotype_definitions_json,
    }
    return render(request, 'jsonl_viewer/search.html', context)


def phenotypes(request):
    stats = PhenotypeStats.objects.select_related('phenotype_definition').all().order_by('-total_microbes')
    top_stats = list(stats[:10])

    # Example: Convert top phenotypes to JSON-friendly format
    top_stats_json = []
    for s in top_stats:
        top_stats_json.append({
            "name": s.phenotype_definition.name,
            "description": s.phenotype_definition.description or "",
            "value_distribution": s.value_distribution or {}
        })

    context = {
        'stats': stats,
        'top_stats': top_stats,
        'top_stats_json': json.dumps(top_stats_json),
    }
    return render(request, 'jsonl_viewer/phenotypes.html', context)

def browse_microbes(request):
    # Aggregate statistics
    total_species_with_ground_truth = Microbe.objects.filter(
        phenotype__value__isnull=False
    ).distinct().count()
    total_species_with_predictions = Microbe.objects.filter(
        predictions__isnull=False
    ).distinct().count()
    total_phenotypes = PhenotypeDefinition.objects.count()
    total_models = Prediction.objects.values('model').distinct().count()

    # Initialize the microbes queryset
    microbes_list = Microbe.objects.all()

    # Pagination
    paginator = Paginator(microbes_list, 20)  # Show 20 microbes per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Get the IDs of the microbes on the current page
    microbe_ids = [microbe.id for microbe in page_obj.object_list]

    # Now get a QuerySet of these microbes with prefetch_related
    microbes = Microbe.objects.filter(id__in=microbe_ids).prefetch_related(
        Prefetch(
            'phenotype_set',
            queryset=Phenotype.objects.exclude(value__in=[None, '', 'N/A']).select_related('definition')
        ),
        Prefetch(
            'predictions__predicted_phenotypes',
            queryset=PredictedPhenotype.objects.exclude(value__in=[None, '', 'N/A']).select_related('definition')
        ),
        Prefetch('descriptions')  # Prefetch MicrobeDescription
    )

    phenotype_definitions = list(PhenotypeDefinition.objects.all())

    # Build a dict of microbes by id for easy access
    microbe_dict = {microbe.id: microbe for microbe in microbes}

    # Process each microbe and add required attributes
    for microbe in microbes:
        # Ground truth phenotypes
        gt_phenotypes = {p.definition.id: p for p in microbe.phenotype_set.all()}
        microbe.ground_truth_count = len(gt_phenotypes)

        # Predicted phenotypes
        predicted_phenotypes = {}
        for prediction in microbe.predictions.all():
            for pp in prediction.predicted_phenotypes.all():
                predicted_phenotypes[pp.definition.id] = pp

        microbe.additional_predictions_count = len(set(predicted_phenotypes.keys()) - set(gt_phenotypes.keys()))

        # Phenotypes availability
        phenotypes_available = []
        for pd in phenotype_definitions:
            if pd.name == "Member of WA subset":
                continue  # Skip this phenotype as per your logic
            has_gt = pd.id in gt_phenotypes
            has_prediction = pd.id in predicted_phenotypes
            phenotypes_available.append({
                'definition': pd,
                'has_gt': has_gt,
                'has_prediction': has_prediction,
            })
        microbe.phenotypes_available = phenotypes_available

    # Ensure the microbes are in the same order as in the page_obj
    processed_microbes = [microbe_dict[microbe_id] for microbe_id in microbe_ids]

    # Replace page_obj.object_list with the processed microbes
    page_obj.object_list = processed_microbes

    context = {
        'page_obj': page_obj,
        'phenotype_definitions': phenotype_definitions,
        'total_species_with_ground_truth': total_species_with_ground_truth,
        'total_species_with_predictions': total_species_with_predictions,
        'total_phenotypes': total_phenotypes,
        'total_models': total_models,
    }

    return render(request, 'jsonl_viewer/browse.html', context)

def report_error(request):
    if request.method == 'POST':
        description = request.POST.get('description', '').strip()
        microbe_id = request.POST.get('microbe_id')
        current_page = request.POST.get('current_page', '')

        if not description:
            messages.error(request, "Error description cannot be empty.")
            return redirect(current_page or 'index')

        try:
            microbe = Microbe.objects.get(id=microbe_id)
        except Microbe.DoesNotExist:
            microbe = None

        ErrorReport.objects.create(
            microbe=microbe,
            description=description
        )

        messages.success(request, "Thank you for reporting the error. Our team will review it.")
        return redirect(current_page or 'index')
    else:
        # For GET requests, redirect to home or appropriate page
        return redirect('index')

# ----- User Authentication Views -----

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful.")
            return redirect('home')
        else:
            messages.error(request, "Unsuccessful registration. Invalid information.")
    else:
        form = CustomUserCreationForm()
    return render(request, 'jsonl_viewer/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'jsonl_viewer/login.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('index')

# ----- Home View for Logged-in Users -----
@login_required
def home(request):
    # Fetch starred microbes
    starred_microbes = request.user.profile.starred_microbes.all()
    
    starred_microbes_with_changes = []
    
    for microbe in starred_microbes:
        # Get the most recent change with non-empty change_description
        recent_change = MicrobeDescription.objects.filter(
            microbe=microbe,
            change_description__isnull=False,  # Ensure change_description exists
            change_description__gt=''  # Ensure it's not empty
        ).order_by('-inference_date_time').first()
        
        starred_microbes_with_changes.append({
            'microbe': microbe,
            'recent_change': recent_change
        })

    context = {
        'starred_microbes_with_changes': starred_microbes_with_changes,
    }

    return render(request, 'jsonl_viewer/home.html', context)

@login_required  # Optional: Restrict access to logged-in users
def download_database(request):
    # Create the HttpResponse object with CSV headers.
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="database_export.csv"'

    writer = csv.writer(response)

    # Write Taxonomy Data
    writer.writerow(['Taxonomy ID', 'Superkingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
    for taxonomy in Taxonomy.objects.all():
        writer.writerow([
            taxonomy.id,
            taxonomy.superkingdom,
            taxonomy.phylum,
            taxonomy.class_name,
            taxonomy.order,
            taxonomy.family,
            taxonomy.genus,
            taxonomy.species
        ])

    # Write Microbe Data
    writer.writerow([])
    writer.writerow(['Microbe ID', 'Binomial Name', 'NCBI ID', 'Taxonomy ID', 'Alternative Names', 'FTP Path', 'Fasta File', 'Access Count Users', 'Access Count Tools'])
    for microbe in Microbe.objects.all():
        writer.writerow([
            microbe.id,
            microbe.binomial_name,
            microbe.ncbi_id,
            microbe.taxonomy.id,
            ', '.join(microbe.alternative_names),
            microbe.ftp_path,
            microbe.fasta_file,
            microbe.access_count_users,
            microbe.access_count_tools
        ])

    # Write PhenotypeDefinition Data
    writer.writerow([])
    writer.writerow(['PhenotypeDefinition ID', 'Name', 'Data Type', 'Allowed Values', 'Description'])
    for pd in PhenotypeDefinition.objects.all():
        writer.writerow([
            pd.id,
            pd.name,
            pd.data_type,
            ', '.join(pd.allowed_values_list),
            pd.description
        ])

    # Write Phenotype Data
    writer.writerow([])
    writer.writerow(['Phenotype ID', 'Microbe ID', 'Definition ID', 'Value'])
    for phenotype in Phenotype.objects.all():
        writer.writerow([
            phenotype.id,
            phenotype.microbe.id,
            phenotype.definition.id,
            json.dumps(phenotype.value)  # Serialize JSONField
        ])

    # Write Prediction Data
    writer.writerow([])
    writer.writerow(['Prediction ID', 'Microbe ID', 'Prediction Identifier', 'Model', 'Inference DateTime'])
    for prediction in Prediction.objects.all():
        writer.writerow([
            prediction.id,
            prediction.microbe.id,
            prediction.prediction_id,
            prediction.model,
            prediction.inference_date_time.strftime('%Y-%m-%d %H:%M:%S')
        ])

    # Write PredictedPhenotype Data
    writer.writerow([])
    writer.writerow(['PredictedPhenotype ID', 'Prediction ID', 'Definition ID', 'Value'])
    for pp in PredictedPhenotype.objects.all():
        writer.writerow([
            pp.id,
            pp.prediction.id,
            pp.definition.id,
            json.dumps(pp.value)  # Serialize JSONField
        ])

    # Write ModelRanking Data
    writer.writerow([])
    writer.writerow(['ModelRanking ID', 'Model', 'Target', 'Balanced Accuracy', 'Precision', 'Sample Size'])
    for mr in ModelRanking.objects.all():
        writer.writerow([
            mr.id,
            mr.model,
            mr.target,
            mr.balanced_accuracy,
            mr.precision,
            mr.sample_size
        ])

    # Write MicrobeDescription Data
    writer.writerow([])
    writer.writerow(['MicrobeDescription ID', 'Microbe ID', 'Description Type', 'Model', 'Description', 'Inference DateTime'])
    for md in MicrobeDescription.objects.all():
        writer.writerow([
            md.id,
            md.microbe.id,
            md.description_type,
            md.model,
            md.description,
            md.inference_date_time.strftime('%Y-%m-%d %H:%M:%S')
        ])

    # Write ErrorReport Data
    writer.writerow([])
    writer.writerow(['ErrorReport ID', 'Microbe ID', 'Description', 'Created At'])
    for er in ErrorReport.objects.all():
        writer.writerow([
            er.id,
            er.microbe.id,
            er.description,
            er.created_at.strftime('%Y-%m-%d %H:%M:%S')
        ])

    return response

