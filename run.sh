#!/bin/bash

if [ "$#" -eq 1 ]
then
    gcloud ml-engine jobs submit training "$1" \
        --module-name=trainer.trainer \
        --package-path=./trainer \
        --job-dir=gs://cs2xb3_ml/"$1" \
        --region=us-east1 \
        --config=trainer/cloudml-gpu.yaml \
        -- \
        --dataset_url=gs://cs2xb3_ml/path/to/dataset \

else
    echo "Must include JOB ID"
fi