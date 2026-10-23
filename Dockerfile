# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code and trained models
COPY src/ ./src/
COPY models/ ./models/

# Expose the API port
EXPOSE 8000

# Command to start the server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]