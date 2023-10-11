# Install Python 3.8.12
FROM python:3.8.12-buster

# Get updates
RUN apt-get update

# Add requirements file to container
ADD ./src/requirements.txt requirements.txt

# Environment setup
RUN pip install -r requirements.txt
RUN pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch-cluster==1.6.0
RUN pip install torch-scatter==2.0.9
RUN pip install torch-sparse==0.6.12
RUN pip install torch-geometric==2.0.4

# Installation of own package
ADD ./ /pathway
WORKDIR /pathway
RUN pip install -e .

# Run pytest

# Execute pytest
RUN pytest ./tests
