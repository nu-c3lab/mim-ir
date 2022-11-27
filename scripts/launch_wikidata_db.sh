echo 'Launching Wikidata Database'
psql -d wikidata -h localhost -p 5432 -U doadmin -w -f data/wikidata_postgresql-dump.sql
