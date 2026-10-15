from django import forms


class PlainNumericInputsMixin:
    """Keep Django numeric validation while removing browser number controls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            if isinstance(field.widget, forms.NumberInput):
                attrs = dict(field.widget.attrs)
                for name in ("min", "max", "step"):
                    attrs.pop(name, None)
                attrs["inputmode"] = "decimal" if isinstance(field, (forms.FloatField, forms.DecimalField)) else "numeric"
                field.widget = forms.TextInput(attrs=attrs)
