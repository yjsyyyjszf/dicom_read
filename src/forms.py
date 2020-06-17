from __future__ import unicode_literals
from django import forms
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm

class LoginForm(AuthenticationForm):
    """
    Bootstraped login form.
    """

    def __init__(self, *args, **kwargs):
        super(LoginForm, self).__init__(*args, **kwargs)

        self.fields['username'].widget.attrs['placeholder'] = ''
        self.fields['password'].widget.attrs['placeholder'] = ''


class BootstrapMixin(forms.BaseForm):
    def __init__(self, *args, **kwargs):
        super(BootstrapMixin, self).__init__(*args, **kwargs)

        for field_name, field in self.fields.items():
            css = field.widget.attrs.get('class', '')
            field.widget.attrs['class'] = ' '.join(
                [css, 'form-control']).strip()

            if field.required:
                field.widget.attrs['required'] = 'required'
            if 'placeholder' not in field.widget.attrs:
                field.widget.attrs['placeholder'] = field.label


class UserPasswordChangeForm(PasswordChangeForm):
    pass

