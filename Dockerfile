# Use an NVIDIA base image with CUDA and cuDNN
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Set the environment variable to non-interactive (this prevents the prompt)
ENV DEBIAN_FRONTEND=noninteractive

# Set the timezone (you can change 'Pacific/Honolulu' to your specific time zone)
ENV TZ=Pacific/Honolulu

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    libgl1-mesa-glx  # This line is new - it installs the required libraries for OpenCV

# Add deadsnakes PPA to get Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa -y

# Install Python 3.9, distutils, and libpq-dev (needed for psycopg2)
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    python3.9-distutils \
    libpq-dev  # This line is new, it installs necessary PostgreSQL libraries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Update alternatives to set python3.9 as the default python3 interpreter
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# It's a good practice to use a fixed version of Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip for Python 3.9
RUN python -m pip install --upgrade pip

# Install specific versions of torch and torchvision with GPU support
# Ensure these versions are compatible with the CUDA version in the image
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Install other Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install 'uvicorn[standard]'

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define any environment variables, if needed
ENV NAME World

# Run main.py when the container launches
CMD ["python", "./main_docker.py"]
