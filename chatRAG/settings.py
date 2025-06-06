"""
Django settings for chatRAG project.

Generated by 'django-admin startproject' using Django 5.1.6.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.1/ref/settings/
"""
import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-n^z-$+@8i1!jybs(shs4(hz=egyrgo##i@wdss&7m$-kr)7%q$'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'logius',
]

# Add this to your chatRAG/settings.py

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
     'formatters': {
        'json': {
            # Format optimized for Promtail parsing the 'message' field as JSON
            'format': '{"timestamp":"%(asctime)s", "level":"%(levelname)s", "logger_name":"%(name)s", "message": %(message)s }',
            'datefmt': '%Y-%m-%dT%H:%M:%S%z',
            'validate': False # Add this for robustness if format causes issues
        },
        'simple_console': {
            'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', # Adjusted for alignment
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    # --- HANDLERS SECTION ---
    'handlers': {
        'console': {  # Handler for simple text output
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple_console',
        },
        # --->>> Ensure this handler definition exists and is spelled correctly <<<---
        'json_console': {  # Handler specifically for JSON output
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'json',  # Uses the 'json' formatter defined above
        },
    },
    'loggers': {
        'django': { # General Django logs
            'handlers': ['console'], # Use simple output
            'level': 'INFO',
            'propagate': False,
        },
        'django.request': { # HTTP request logs (often noisy)
            'handlers': ['console'], # Use simple output
            'level': 'WARNING', # Quieter level
            'propagate': False,
        },
        'rag_metrics': { # Your metrics logger
            'handlers': ['json_console'], # --->>> Reference the JSON handler <<<---
            'level': 'INFO',
            'propagate': False,
        },
        'user_feedback': { # Your feedback logger
            'handlers': ['json_console'], # --->>> Reference the JSON handler <<<---
            'level': 'INFO',
            'propagate': False,
        },
        'logius': { # Your main app logger (from getLogger(__name__))
            'handlers': ['console'], # Use simple output
            'level': 'INFO', # Or 'DEBUG' if needed
            'propagate': False,
        },
         # Optional root logger configuration (catches logs not otherwise configured)
        # '': {
        #     'handlers': ['console'],
        #     'level': 'WARNING', # Default level for unconfigured loggers
        # },
    },
}

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'chatRAG.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates']
        ,
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'chatRAG.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')

DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql', # Use the PostgreSQL engine
            'NAME': 'logius-standaarden',                          # Database name from .env
            'USER': 'postgres',                          # Database user from .env
            'PASSWORD': 'postgres',                  # Database password from .env
            'HOST': 'db',                          # Hostname from .env (should be 'db' for docker-compose)
            'PORT': '5432',                          # Port from .env (should be '5432')
        }
    }


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
