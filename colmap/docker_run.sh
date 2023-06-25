#!/bin/bash

docker run -u $(id -u):$(id -g) --rm -it \
 -v $1:/data:rw \
 --ipc=host \
 --net=host \
 -e DISPLAY \
  colmap
