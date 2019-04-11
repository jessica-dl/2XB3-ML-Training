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
        --dataset_path=gs://cs2xb3_ml/images_ages.hdf5 \
        --generator_weights=gs://cs2xb3_ml/generator-1000000.h5 \
        --discriminator_weights=gs://cs2xb3_ml/discriminator-1000000.h5

else
    echo "Must include JOB ID"
fi