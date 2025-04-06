from django import forms
from .models import Payment
from django.contrib.auth import get_user_model

User = get_user_model()
# Create a form for the Payment model
class PaymentForm(forms.ModelForm):
    patient = forms.CharField(
        label='Patient',
        max_length=150,
        widget=forms.TextInput(attrs={'placeholder': 'Patient Name', 'readonly': True})
    )
    amount = forms.IntegerField(label='Appoint Fee(Rs)', min_value=1, max_value=1500, widget=forms.NumberInput(attrs={'placeholder': 'Amount in Rs'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'placeholder': 'Confirm your Email'}))

    class Meta:
        model = Payment
        fields = ["patient", "amount", "email"]

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            patient = f"{user.first_name} {user.last_name}"
            self.initial['patient'] = patient
            self.initial['email'] = user.email  
            self.initial['amount'] = 120 
            self.fields['patient'].widget.attrs['readonly'] = True
            self.fields['amount'].widget.attrs['readonly'] = True
            self.fields['email'].widget.attrs['readonly'] = True  