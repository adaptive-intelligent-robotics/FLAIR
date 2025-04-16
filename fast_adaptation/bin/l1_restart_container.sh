#!/bin/bash

if [ $# -lt 1 ]
then
    echo "Must pass in container to restart"
    exit 1
fi
container=$1

# Launch the containers
docker kill ${container}
docker rm ${container}

# Set up variables that are local to a given Orin/Bot
export CAMERA_FORWARD=39004209
export INFLUXDB_KEY="XXXXXXXX"
export INFLUXDB_BUCKET="paper_bucket"

docker-compose -f docker_compose_l1.yaml --project-directory ./ up -d -- ${container}
