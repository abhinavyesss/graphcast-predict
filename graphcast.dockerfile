# Get ubuntu image from https://hub.docker.com/_/ubuntu.
FROM ubuntu:22.04

# Update ubuntu's package installer and download python and other important packages.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3 python3-pip curl unzip sudo

# Download components required by Graphcast.
RUN pip3 install --upgrade https://github.com/deepmind/graphcast/archive/master.zip

# Create dashboard directory.
RUN mkdir -p /app
WORKDIR /app

# Copy all files to environment.
COPY . .

# Download necessary packages.
RUN sudo apt-get install -y libgeos-dev
RUN pip3 uninstall -y shapely
RUN pip3 install -r requirements.txt
RUN pip3 install shapely --no-binary shapely
RUN pip3 install -U "jax[cuda12_pip]" -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

# Move the CDS key to the root folder.
RUN mv /app/.cdsapirc /root/.cdsapirc