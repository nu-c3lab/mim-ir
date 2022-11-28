echo 'Initializing Wikidata Database'
createuser -h localhost -p 5432 -ds doadmin
createdb -h localhost -p 5432 -U doadmin wikidata
psql -d wikidata -h localhost -p 5432 -U doadmin -w -f data/wikidata_postgresql-dump.sql
echo 'Done!'
