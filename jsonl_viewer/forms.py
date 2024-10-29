from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, help_text='Required. Enter a valid email address.')
    institute = forms.CharField(max_length=255, required=False, help_text='Optional. Enter your affiliated institute.')

    class Meta:
        model = User
        fields = ('username', 'email', 'institute', 'password1', 'password2')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            # Create or update the Profile
            Profile.objects.update_or_create(user=user, defaults={'institute': self.cleaned_data.get('institute', '')})
        return user