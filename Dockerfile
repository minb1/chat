
# Use the official Python 3.10 slim image as a base for a smaller footprint
FROM python:3.10-slim as base

# Prevents Python from writing pyc files to disc and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required to build some Python packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Django runs on
EXPOSE 8000

# Use an entrypoint to run migrations before starting the server
CMD ["sh", "-c", "python manage.py makemigrations logius && python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
