from django import template
from django.template.defaultfilters import stringfilter
from collections import defaultdict
from jsonl_viewer.models import PhenotypeSummary
from jsonl_viewer.utils import (
    calculate_contrasting_color as consistent_color,
    normalize_value,
    clean_phenotype,
    clean_quotes
)
import re
import ast
import hashlib
import colorsys
import markdown
import bleach
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def multiply(value, arg):
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def map_predictions(microbe, phenotype_def):
    """Get prediction summary for a microbe and phenotype definition"""
    try:
        summary = PhenotypeSummary.objects.get(
            microbe=microbe,
            definition=phenotype_def
        )
        return {
            'majority_value': summary.majority_value,
            'agreement_percentage': summary.agreement_percentage,
            'supporting_models': summary.supporting_models,
            'total_models': summary.total_models
        }
    except PhenotypeSummary.DoesNotExist:
        return {
            'majority_value': None,
            'agreement_percentage': 0,
            'supporting_models': 0,
            'total_models': 0
        }

@register.filter(name='sum_species_count')
def sum_species_count(phyla_dict):
    return sum(len(microbes) for microbes in phyla_dict)

@register.filter
@stringfilter
def markdownify(text):
    """
    Converts Markdown text to sanitized HTML, including custom highlighting.
    """
    if not text:
        return ""

    # Replace __text__ with <mark> for highlighting
    text = re.sub(r'__(.*?)__', r'<mark>\1</mark>', text)

    # Convert frozenset to list before adding new tags
    allowed_tags = list(bleach.sanitizer.ALLOWED_TAGS) + ['mark']
    allowed_attributes = bleach.sanitizer.ALLOWED_ATTRIBUTES

    # Convert Markdown to HTML with desired extensions
    html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
    
    # Sanitize the HTML output
    clean_html = bleach.clean(
        html,
        tags=allowed_tags,
        attributes=allowed_attributes,
        strip=True
    )
    
    return mark_safe(clean_html)

@register.filter(name='to_float')
def to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0

@register.filter(name='consistent_color')
def consistent_color_filter(value):
    return consistent_color(value)

@register.filter(name='normalize_value')
def normalize_value_filter(value):
    return normalize_value(value)

@register.filter(name='reject')
def reject(value, arg):
    if not isinstance(value, (list, tuple)):
        return value
    if arg == 'not':
        return [item for item in value if item]
    else:
        return [item for item in value if not eval(f"item {arg}")]

@register.filter(name='clean_alternative_names')
def clean_alternative_names(value):
    try:
        names_list = ast.literal_eval(value)
        if isinstance(names_list, list):
            return ", ".join(filter(bool, names_list))
    except (ValueError, SyntaxError):
        pass
    return ""

@register.filter
@stringfilter
def make_subtle(value):
    """
    Make a color more subtle by reducing its saturation and increasing its lightness.
    """
    # Remove the '#' if present
    value = value.lstrip('#')
    
    # Convert hex to RGB
    rgb = tuple(int(value[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(*[x/255.0 for x in rgb])
    
    # Reduce saturation and increase lightness
    s = max(0, s - 0.3)  # Reduce saturation by 0.3
    l = min(1, l + 0.3)  # Increase lightness by 0.3
    
    # Convert back to RGB
    rgb = colorsys.hls_to_rgb(h, l, s)
    
    # Convert RGB back to hex
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

@register.filter(name='clean_phenotype')
def clean_phenotype_filter(value):
    return clean_phenotype(value)

@register.filter(name='clean_quotes')
def clean_quotes_filter(value):
    return clean_quotes(value)

@register.filter
def sort_predictions(predictions, na_counts):
    """
    Sorts prediction models based on the number of 'N/A' values (ascending).
    Models with fewer 'N/A's come first.
    """
    return sorted(predictions, key=lambda x: na_counts.get(x.model, 0))