# Step 1: Use an official Python runtime as a base image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory (with app.py and requirements.txt) into the container
COPY . /app

# Step 4: Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the Streamlit default port (8501)
EXPOSE 8501

# Step 6: Define the command to run Streamlit when the container starts
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
