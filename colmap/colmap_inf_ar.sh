#!/bin/bash

# The project folder must contain a folder "images" with all the images.
DATASET_PATH=/database/colmap_test

colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images