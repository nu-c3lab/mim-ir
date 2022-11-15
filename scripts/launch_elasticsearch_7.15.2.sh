if curl -I localhost:9200 2>/dev/null; then
    exit
fi
cd elasticsearch-7.15.2
bin/elasticsearch 2>&1 >/dev/null &
while ! curl -I localhost:9200 2>/dev/null;
do
  sleep 2;
done