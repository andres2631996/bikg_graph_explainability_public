#!/bin/bash
ml gcccuda
pip install -r src/requirements.txt
pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-cluster==1.6.0 
pip install torch-scatter==2.0.9
pip install torch-sparse==0.6.12
pip install torch-geometric==2.0.4
pip install -e .
