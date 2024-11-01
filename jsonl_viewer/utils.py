import hashlib
from collections import Counter
from django.template.defaultfilters import stringfilter
import re
import ast
import colorsys

def hash_value(value):
    """
    Generates an integer hash for a given value.
    """
    normalized_value = str(value).lower().replace('"', '').replace(' ', '').strip()
    hash_object = hashlib.md5(normalized_value.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return hash_int

def calculate_contrasting_color(value):
    """
    Generates a consistent HSL color based on the input value.
    """
    if value is None or str(value).strip().lower() == 'n/a':
        return '#F0F0F0'  # Light grey
    hash_int = hash_value(value)
    hue = hash_int % 360
    saturation = 50  # Reduced saturation for subtler colors
    lightness = 80  # Increased lightness for softer backgrounds
    return f"hsl({hue}, {saturation}%, {lightness}%)"

def normalize_value(value):
    """
    Normalizes a value for comparison by stripping and lowering.
    """
    if isinstance(value, str):
        return value.strip().lower()
    elif value is not None:
        return str(value).strip().lower()
    return ''

def clean_phenotype(value):
    """
    Cleans the phenotype string by removing unwanted characters or substrings.
    """
    if not isinstance(value, str):
        return value
    # Remove parentheses and their content
    cleaned = re.sub(r'\(.*?\)', '', value)
    # Remove surrounding quotes
    cleaned = re.sub(r'^["\']|["\']$', '', cleaned)
    return cleaned.strip()

def clean_quotes(value):
    """
    Removes surrounding quotation marks from a string.
    """
    if not isinstance(value, str):
        return value
    return value.strip('"').strip("'")
