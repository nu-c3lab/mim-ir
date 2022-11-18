echo "Downloading Elasticsearch..."
bash scripts/download_elastic_7.15.2.sh

echo "NOTE: we set jvm options -Xms and -Xmx for Elasticsearch to be 4GB"
echo "We suggest you set them as large as possible in: elasticsearch-7.15.2/config/jvm.options"
cp jvm.options elasticsearch-7.15.2/config/jvm.options

echo "Downloading wikipedia source documents..."
bash scripts/download_processed_wiki.sh

echo "Running Elasticsearch and indexing Wikipedia documents..."
bash scripts/launch_elasticsearch_7.15.2.sh
python -m scripts.index_processed_wiki
