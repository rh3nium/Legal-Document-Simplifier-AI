# Use a Python 3.9 base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app

# Expose the port your Flask app runs on
EXPOSE 7860

# Define the command to run your application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]

