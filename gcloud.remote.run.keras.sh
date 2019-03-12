#!/bin/bash

gcloud ml-engine jobs submit training JOB2 \
    --module-name=trainer.cnn_with_keras \
    --package-path=./trainer \
    --job-dir=gs://cs2xb3_ml \
    --region=us-central1 \
    --config=trainer/cloudml-gpu.yaml


    #--scale-tier CUSTOM \
    #--master-type standard_gpu \
    #--runtime-version 1.13 \
    #--python-version 3.5