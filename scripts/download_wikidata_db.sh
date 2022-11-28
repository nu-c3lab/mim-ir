echo 'Downloading Wikidata Database snapshot'
# download the wiki dump file
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18-lc_ujro149pWer92hPkUH-4FADbX6U' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18-lc_ujro149pWer92hPkUH-4FADbX6U" -O data/wikidata_postgresql-dump.tar.bz2 && rm -rf /tmp/cookies.txt
# verify that we have the whole thing
cd data
echo 'Extracting Wikidata Database snapshot'
tar -xjf wikidata_postgresql-dump.tar.bz2
# clean up
rm wikidata_postgresql-dump.tar.bz2
echo 'Done!'
