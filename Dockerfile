# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirement files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into container
COPY . .

# Default command (can be changed later)
CMD ["python", "src/ingest.py", "--help"]
