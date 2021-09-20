#!/bin/sh
# When the Mindmeld container starts, there is a chance that Mindmeld might try to interact with the ElasticSearch cluster before it's ready
# This script waits for the cluster to be in a healthy stat before finishing, allowing the Docker container to not try to access the cluster too early

echo "Waiting For ElasticSearch Cluster To Be Available..."

until $(curl -s -XGET 'http://localhost:9200/_cluster/health?wait_for_status=green' > /dev/null); do
    printf 'ElasticSearch cluster not in green state, trying again in 5 seconds... \n'
    sleep 5
done

echo "ElasticSearch Cluster In Green State; Continuing..."