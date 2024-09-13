from django import template
import math
import hashlib

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
    # Normalize the string: convert to lowercase, remove quotes and spaces
    normalized_value = str(value).lower().replace('"', '').replace(' ', '').strip()
    # Use a consistent hashing method
    hash_object = hashlib.md5(normalized_value.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return hash_int

@register.filter(name='consistent_color')
def consistent_color(value):
    # Generate a unique color based on the hash of the value
    hash_int = hash_value(value)
    # Map the hash value to a hue value between 0 and 360
    hue = hash_int % 360
    # Return the color in HSL format
    return f"hsl({hue}, 70%, 80%)"

@register.filter(name='normalize_value')
def normalize_value(value):
    return str(value).lower().replace('"', '').replace(' ', '').strip()
