
from datetime import date

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

from .models import UserProfile


# ============================================================
# REGISTRATION FORM: ACCOUNT + HEALTH PROFILE
# ============================================================

class RegisterForm(UserCreationForm):

    first_name = forms.CharField(
        max_length=100,
        label="First name",
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "Enter your first name",
            "autocomplete": "given-name",
        }),
    )

    last_name = forms.CharField(
        max_length=100,
        label="Last name",
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "Enter your last name",
            "autocomplete": "family-name",
        }),
    )

    email = forms.EmailField(
        label="Email address",
        widget=forms.EmailInput(attrs={
            "class": "form-control",
            "placeholder": "you@example.com",
            "autocomplete": "email",
        }),
    )

    phone = forms.CharField(
        required=False,
        max_length=15,
        label="Phone number",
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "Enter your 10-digit phone number",
            "type": "tel",
            "autocomplete": "tel",
            "pattern": "[0-9]{10}",
            "title": "Enter a 10-digit phone number",
            "inputmode": "numeric",
        }),
    )

    username = forms.CharField(
        label="Username",
        min_length=3,
        max_length=150,
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "Choose a username",
            "autocomplete": "username",
        }),
    )

    # -------------------- HEALTH INFORMATION --------------------

    date_of_birth = forms.DateField(
        required=False,
        label="Date of birth",
        widget=forms.DateInput(attrs={
            "type": "date",
            "class": "form-control",
        }),
    )

    gender = forms.ChoiceField(
        required=False,
        label="Gender",
        choices=[
            ("", "Select gender"),
            *UserProfile.GENDER_CHOICES,
        ],
        widget=forms.Select(attrs={
            "class": "form-select",
        }),
    )

    occupation = forms.CharField(
        required=False,
        max_length=150,
        label="Occupation",
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "e.g. Student",
        }),
    )

    height_cm = forms.FloatField(
        required=False,
        label="Height (cm)",
        min_value=50,
        max_value=250,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "placeholder": "e.g. 165",
            "step": "0.1",
        }),
    )

    weight_kg = forms.FloatField(
        required=False,
        label="Weight (kg)",
        min_value=20,
        max_value=500,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "placeholder": "e.g. 60",
            "step": "0.1",
        }),
    )

    smoking_status = forms.ChoiceField(
        required=False,
        label="Smoking status",
        choices=UserProfile.SMOKING_STATUS_CHOICES,
        initial="unknown",
        widget=forms.Select(attrs={
            "class": "form-select",
        }),
    )

    # -------------------- HEALTH HISTORY --------------------

    diabetes_history = forms.ChoiceField(
        required=False,
        label="History of diabetes",
        choices=UserProfile.YES_NO_UNKNOWN,
        initial="unknown",
        widget=forms.Select(attrs={
            "class": "form-select",
        }),
    )

    family_diabetes_history = forms.ChoiceField(
        required=False,
        label="Family history of diabetes",
        choices=UserProfile.YES_NO_UNKNOWN,
        initial="unknown",
        widget=forms.Select(attrs={
            "class": "form-select",
        }),
    )

    heart_disease_history = forms.ChoiceField(
        required=False,
        label="History of heart disease",
        choices=UserProfile.YES_NO_UNKNOWN,
        initial="unknown",
        widget=forms.Select(attrs={
            "class": "form-select",
        }),
    )

    # -------------------- PASSWORDS --------------------

    password1 = forms.CharField(
        label="Password",
        strip=False,
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Create a password",
            "autocomplete": "new-password",
            "minlength": "8",
        }),
    )

    password2 = forms.CharField(
        label="Confirm password",
        strip=False,
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Confirm your password",
            "autocomplete": "new-password",
        }),
    )

    class Meta:
        model = User
        fields = [
            "first_name",
            "last_name",
            "username",
            "email",
            "password1",
            "password2",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for field in self.fields.values():
            field.widget.attrs.setdefault("class", "form-control")

        self.fields["username"].widget.attrs.update({
            "minlength": "3",
        })

        self.fields["password1"].widget.attrs.update({
            "minlength": "8",
            "autocomplete": "new-password",
        })

        self.fields["password2"].widget.attrs.update({
            "autocomplete": "new-password",
        })

    # -------------------- EMAIL VALIDATION --------------------

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()

        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError(
                "An account with this email already exists."
            )

        return email

    # -------------------- PHONE VALIDATION --------------------

    def clean_phone(self):
        phone = self.cleaned_data.get("phone", "").strip()

        if phone and not phone.isdigit():
            raise forms.ValidationError(
                "Enter a phone number using digits only."
            )

        if phone and len(phone) != 10:
            raise forms.ValidationError(
                "Enter a valid 10-digit phone number."
            )

        return phone

    # -------------------- DATE OF BIRTH VALIDATION --------------------

    def clean_date_of_birth(self):
        dob = self.cleaned_data.get("date_of_birth")

        if not dob:
            return dob

        today = date.today()

        if dob > today:
            raise forms.ValidationError(
                "Date of birth cannot be in the future."
            )

        age = (
            today.year - dob.year
            - ((today.month, today.day) < (dob.month, dob.day))
        )

        if age < 18:
            raise forms.ValidationError(
                "This risk-screening workflow is intended for adults aged 18 or older."
            )

        if age > 100:
            raise forms.ValidationError(
                "Please enter a valid date of birth."
            )

        return dob

    # -------------------- PASSWORD VALIDATION --------------------

    def clean_password1(self):
        password = self.cleaned_data.get("password1")

        if password and len(password) < 8:
            raise forms.ValidationError(
                "Password must contain at least 8 characters."
            )

        if password and not any(char.isalpha() for char in password):
            raise forms.ValidationError(
                "Password must contain at least one letter."
            )

        if password and not any(char.isdigit() for char in password):
            raise forms.ValidationError(
                "Password must contain at least one number."
            )

        return password

    # -------------------- SAVE USER AND HEALTH PROFILE --------------------

    def save(self, commit=True):
        user = super().save(commit=commit)

        if not commit:
            return user

        profile_fields = [
            "date_of_birth",
            "gender",
            "occupation",
            "height_cm",
            "weight_kg",
            "smoking_status",
            "diabetes_history",
            "family_diabetes_history",
            "heart_disease_history",
        ]

        profile_data = {
            field: self.cleaned_data.get(field)
            for field in profile_fields
            if self.cleaned_data.get(field) not in (None, "")
        }

        profile_data["phone"] = self.cleaned_data.get("phone", "")

        UserProfile.objects.update_or_create(
            user=user,
            defaults=profile_data,
        )

        return user


# ============================================================
# LOGIN FORM
# ============================================================

class LoginForm(AuthenticationForm):

    username = forms.CharField(
        label="Username",
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "Username",
            "autocomplete": "username",
        }),
    )

    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Password",
            "autocomplete": "current-password",
        }),
    )


# ============================================================
# EDIT HEALTH PROFILE FORM
# ============================================================

class UserProfileForm(forms.ModelForm):

    class Meta:
        model = UserProfile

        fields = [
            "date_of_birth",
            "gender",
            "occupation",
            "height_cm",
            "weight_kg",
            "smoking_status",
            "diabetes_history",
            "family_diabetes_history",
            "heart_disease_history",
            "phone",
        ]

        widgets = {
            "date_of_birth": forms.DateInput(attrs={
                "type": "date",
                "class": "form-control",
            }),

            "gender": forms.Select(attrs={
                "class": "form-select",
            }),

            "occupation": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "Occupation",
            }),

            "height_cm": forms.NumberInput(attrs={
                "class": "form-control",
                "min": "50",
                "max": "250",
                "step": "0.1",
            }),

            "weight_kg": forms.NumberInput(attrs={
                "class": "form-control",
                "min": "20",
                "max": "500",
                "step": "0.1",
            }),

            "smoking_status": forms.Select(attrs={
                "class": "form-select",
            }),

            "diabetes_history": forms.Select(attrs={
                "class": "form-select",
            }),

            "family_diabetes_history": forms.Select(attrs={
                "class": "form-select",
            }),

            "heart_disease_history": forms.Select(attrs={
                "class": "form-select",
            }),

            "phone": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "Enter your 10-digit phone number",
                "type": "tel",
                "pattern": "[0-9]{10}",
                "maxlength": "15",
                "inputmode": "numeric",
            }),
        }

    def clean_phone(self):
        phone = self.cleaned_data.get("phone", "").strip()

        if phone and not phone.isdigit():
            raise forms.ValidationError(
                "Enter a phone number using digits only."
            )

        if phone and len(phone) != 10:
            raise forms.ValidationError(
                "Enter a valid 10-digit phone number."
            )

        return phone

    def clean_date_of_birth(self):
        dob = self.cleaned_data.get("date_of_birth")

        if dob:
            today = date.today()

            if dob > today:
                raise forms.ValidationError(
                    "Date of birth cannot be in the future."
                )

            age = (
                today.year - dob.year
                - ((today.month, today.day) < (dob.month, dob.day))
            )

            if age < 18 or age > 100:
                raise forms.ValidationError(
                    "Please enter a valid date of birth for an adult."
                )

        return dob
