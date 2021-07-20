# forms.py
from django import forms
from .models import *


class uploadPictureForm(forms.ModelForm):
    class Meta:
        model = uploadPicture
        fields = ['name', 'Main_Img']

