from django.apps import AppConfig


class LogiusConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "logius"

    def ready(self):
        print("This is the startup command")
