FROM ubuntu:16.04
MAINTAINER Ricardo Rocha <coastrock.github.io>

# Update OS
RUN apt-get update

# Verify version
RUN cat /etc/lsb-release

# Install Python3
RUN apt-get install -y python3

# Install pip for Python3 
RUN apt-get install -y python3-pip

# Install numpy, pandas using pip
RUN pip3 install sklearn matplotlib numpy pandas seaborn

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run wine_data_parser.py when the container launches
CMD ["python3", "/app/wine_data_parser_classifier.py"]

