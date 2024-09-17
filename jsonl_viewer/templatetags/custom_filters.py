from django import template
import math
import hashlib
import ast
from django.template.defaultfilters import stringfilter
import colorsys
import re
from collections import defaultdict
import markdown
import bleach
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def get(dictionary, key):
    return dictionary.get(key)

@register.filter(name='to_float')
def to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

@register.filter(name='subtract')
def subtract(value, arg):
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter(name='abs')
def abs_filter(value):
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return 0

@register.filter(name='modulo')
def modulo(num, val):
    try:
        return int(num) % int(val)
    except (ValueError, TypeError):
        return 0

@register.filter(name='hash_value')
def hash_value(value):
    normalized_value = str(value).lower().replace('"', '').replace(' ', '').strip()
    hash_object = hashlib.md5(normalized_value.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return hash_int

@register.filter(name='consistent_color')
def consistent_color(value):
    if value is None or str(value).strip().lower() == 'n/a':
        return '#F0F0F0'  # Light grey
    hash_int = hash_value(value)
    hue = hash_int % 360
    saturation = 50  # Reduced saturation for subtler colors
    lightness = 80  # Increased lightness for softer backgrounds
    return f"hsl({hue}, {saturation}%, {lightness}%)"

@register.filter(name='normalize_value')
def normalize_value(value):
    if isinstance(value, str):
        return value.strip().lower()
    elif value is not None:
        return str(value).strip().lower()
    return ''

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
def clean_phenotype(value):
    """
    Cleans the phenotype string by removing unwanted characters or substrings.
    Modify the logic as per your specific requirements.
    """
    if not isinstance(value, str):
        return value
    # Remove parentheses and their content
    cleaned = re.sub(r'\(.*?\)', '', value)
    # Remove surrounding quotes
    cleaned = re.sub(r'^["\']|["\']$', '', cleaned)
    return cleaned.strip()

@register.filter(name='clean_quotes')
def clean_quotes(value):
    """
    Removes surrounding quotation marks from a string.
    """
    if not isinstance(value, str):
        return value
    return value.strip('"').strip("'")

@register.filter
def sort_predictions(predictions, na_counts):
    """
    Sorts prediction models based on the number of 'N/A' values (ascending).
    Models with fewer 'N/A's come first.
    """
    return sorted(predictions, key=lambda x: na_counts.get(x.model, 0))

@register.filter(name='markdownify')
def markdownify(text):
    """
    Converts Markdown text to sanitized HTML.
    """
    if not text:
        return ""
    # Convert Markdown to HTML with desired extensions
    html = markdown.markdown(text, extensions=['extra', 'codehilite', 'toc'])
    
    # Convert ALLOWED_TAGS from frozenset to list
    allowed_tags = list(bleach.sanitizer.ALLOWED_TAGS) + [
        'p', 'pre', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'img', 'span', 'div', 'br', 'hr', 'ul', 'ol', 'li', 'strong', 'em',
        'a', 'blockquote', 'table', 'thead', 'tbody', 'tr', 'th', 'td'
    ]
    
    allowed_attributes = {
        '*': ['class', 'id', 'style'],
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title'],
    }
    
    # Sanitize the HTML
    cleaned_html = bleach.clean(html, tags=allowed_tags, attributes=allowed_attributes)
    
    return mark_safe(cleaned_html)