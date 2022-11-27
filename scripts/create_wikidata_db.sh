echo 'Creating Wikidata Database'
postgres -D /usr/local/var/postgres &
createuser -h localhost -p 5432 -ds doadmin
createdb -h localhost -p 5432 -U doadmin wikidata
echo 'Done!'
