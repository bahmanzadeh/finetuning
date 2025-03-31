# Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Ensure 'python' points to 'python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /app

# Copy requirements.txt (create one with necessary packages)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of application
COPY fine-tune.py .
# Copy the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set the entrypoint to use the script
ENTRYPOINT ["/app/entrypoint.sh"]

