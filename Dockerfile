# Use an official Python runtime as a parent image.
# We choose a slim version to keep the image size small.
FROM python:3.9-slim-buster

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install scikit-learn and numpy first to avoid potential conflicts
RUN pip install --no-cache-dir numpy scikit-learn

# Install any remaining packages specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container.
# The '.' indicates copy everything from the current directory on the host to the WORKDIR in the container.
COPY . .

# Command to run the application when the container starts.
CMD ["python", "hr_assistant.py", "What is the sick leave policy?"]
