# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies: git, curl, ffmpeg
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . .

# Expose the port (adjust to your app, e.g., 8000 for FastAPI)
EXPOSE 8000

# Run the application using Uvicorn (adjust if needed)
CMD ["uvicorn", "app.API:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
