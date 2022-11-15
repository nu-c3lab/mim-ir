import bz2
import html
import json
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser
from elasticsearch import Elasticsearch

def chunks(iterator, n):
    """Yield successive n-sized chunks from iterator."""
    chunk = []
    for x in iterator:
        chunk.append(x)
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

def process_line_beerqa_bz2(line):
    data = json.loads(line)
    docid = f"wiki-{data['id']}"
    item = {'id': data['id'],
            'url': data['url'],
            'title': data['title'],
            'title_unescape': html.unescape(data['title']),
            'text': ''.join(data['text']),
            'doc_text': ''.join(data['text']),
            'original_json': json.dumps({'id': data['id'], 'url': data['url'], 'title': data['title'], 'text': data['text'], 'docidid': docid}), #line,
            'docid': docid,
            'doctype': 'doc',
            }
    # tell elasticsearch we're indexing documents
    yield "{}\n{}".format(json.dumps({ 'index': { '_id': docid, '_routing': docid } }), json.dumps(item))

def generate_indexing_queries_from_beerqa_bz2(bz2file, dry=False):
    if dry:
        return

    with bz2.open(bz2file, 'rt') as f:
        for line in f:
            yield from process_line_beerqa_bz2(line)

def index_chunk(chunk, index):
    es = Elasticsearch(timeout=600)
    res = es.bulk(index=index, doc_type='doc', body='\n'.join(chunk), timeout='600s')
    assert not res['errors'], res
    return len(chunk)

def ensure_index(index, reindex):
    es = Elasticsearch(timeout=600)
    if es.indices.exists(index=index) and reindex:
        print('deleting index...')
        es.indices.delete(index=index)
    if not es.indices.exists(index=index):
        print('creating index...')
        es.indices.create(index=index,
                body=json.dumps({
                    "mappings":{
                        "properties": {
                            "id": {
                                "type": "keyword"
                            },
                            "url": {
                                "type": "keyword"
                            },
                            "docid": {
                                "type": "keyword"
                            },
                            "doctype": {
                                "type": "join",
                                "relations": {
                                    "doc": "para"
                                }
                            },
                            "title": {
                                "type": "text",
                                "analyzer": "simple_bigram_analyzer"
                            },
                            "title_unescape": {
                                "type": "text",
                                "analyzer": "simple_bigram_analyzer"
                            },
                            "text": {
                                "type": "text",
                                "analyzer": "bigram_analyzer"
                            },
                            "doc_text": {
                                "type": "text",
                                "analyzer": "bigram_analyzer",
                                "similarity": "scripted_bm25"
                            },
                            "original_json": {
                                "type": "text"
                            }
                        }
                    },
                    "settings": {
                        "analysis": {
                            "analyzer":{
                                "simple_bigram_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": [
                                            "lowercase", "shingle"
                                    ]
                                },
                                "bigram_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": [
                                        "lowercase", "my_stop", "shingle", "remove_filler", "remove_empty"
                                    ]
                                }
                            },
                            "filter":{
                                "my_stop": {
                                    "type": "stop",
                                    "stopwords": [
                                        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
                                        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
                                        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
                                        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                                        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                                        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
                                        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
                                        "at", "by", "for", "with", "about", "against", "between", "into", "through",
                                        "during", "before", "after", "above", "below", "to", "from", "up", "down",
                                        "in", "out", "on", "off", "over", "under", "again", "further", "then",
                                        "once", "here", "there", "when", "where", "why", "how", "all", "any",
                                        "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
                                        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
                                        "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
                                        "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
                                        "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
                                        "won", "wouldn", "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
                                    ]
                                },
                                "remove_filler": {
                                    "type": "pattern_replace",
                                    "pattern": ".*_.*",
                                    "replace": ""
                                },
                                "remove_empty": {
                                    "type": "stop",
                                    "stopwords": [""]
                                }
                            }
                        },
                        "index": {
                            "similarity": {
                                "scripted_bm25": {
                                    "type": "scripted",
                                    "script": {
                                        "source": "double tf = doc.freq * (1 + 1.2) / (doc.freq + 1.2); double idf = Math.max(0, Math.log((field.docCount-term.docFreq+0.5)/(term.docFreq+0.5))); return query.boost * tf * Math.pow(idf, 2);"
                                    }
                                }
                            }
                        }
                    }
                }))

def query_generator(index_type):
    if index_type == "beerqa":
        filelist = glob('data/enwiki-20200801-pages-articles-tokenized/*/wiki_*.bz2')
        for f in tqdm(filelist, position=1):
            yield from generate_indexing_queries_from_beerqa_bz2(f)

def main(args):
    index = 'beerqa_wiki_docs'

    if not args.dry:
        print(f'Making index "{index}"...')
        ensure_index(index, args.reindex)

        print('Indexing...')

        pbar = tqdm(total=-1, position=0)
        def update(*a):
            pbar.update(a[0])

        pool = Pool()
        chunksize = 8192
        for i, chunk in enumerate(chunks(query_generator(args.type), chunksize)):
            pool.apply_async(index_chunk, [chunk, index], error_callback=print, callback=update)
            #index_chunk(chunk, index)
            #update(len(chunk))

        pool.close()
        pool.join()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--reindex', action='store_true', help="Reindex everything")
    parser.add_argument('--dry', action='store_true', help="Dry run")

    args = parser.parse_args()

    main(args)