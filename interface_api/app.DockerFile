# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file into the container
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory in the container
COPY . .

# Expose port 80 to allow communication to the FastAPI application
EXPOSE 80

# Command to run the FastAPI application with Swagger UI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload", "--root-path", "/api"]
