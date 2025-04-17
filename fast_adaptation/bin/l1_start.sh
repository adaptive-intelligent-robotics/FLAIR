#!/bin/bash
# Set up variables that are local to a given Orin/Bot
export CAMERA_FORWARD=39004209
export INFLUXDB_KEY="q2ptkRdsLF8ezJmr5AvqepNEpYZtMCg8V8KI9eCIkj98fI9rShw3zTILpE93_znE-7dZhT0xeBVhCXcEwB5_ug=="
export INFLUXDB_BUCKET="paper_bucket"

docker-compose -f docker_compose_l1.yaml --project-directory ./ up -d
