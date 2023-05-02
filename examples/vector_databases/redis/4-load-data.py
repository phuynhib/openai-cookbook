from app import env, arguments
import openai
import os
import sys
import numpy as np
import pandas as pd
from typing import List

# use helper function in nbutils.py to download and read the data
# this should take from 5-10 min to run
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import nbutils

# nbutils.download_wikipedia_data()
# data = nbutils.read_wikipedia_data()

arguments.add(
    '--pickle', type=str, required=True, default=None,
    help='pickle file to load'
)
args = arguments.all()


# args.pickle = 'data.pkl'
# pickle_file = '/var/www/html/medscape/medscape-content-feed/linear_model/storage/shared/df_cache/EMBEDDINGS-10000/EMBEDDINGS-10000_20230329101648.pkl'
# pickle_file = '/var/www/html/medscape/medscape-content-feed/linear_model/storage/shared/df_cache/EMBEDDINGS-10000-with-titles/EMBEDDINGS-10000-with-titles_20230331110930.pkl'
# pickle_file = '/home/phu/htdocs/medscape/medscape-content-feed/linear_model/storage/shared/df_cache/EMBEDDINGS-ALL-2015-to-2016/EMBEDDINGS-ALL-2015-to-2016_20230330170354.pkl'
# pickle_file = '/var/www/html/medscape/medscape-content-feed/linear_model/storage/shared/df_cache/EMBEDDINGS-ALL-2015-to-2016-with-titles/EMBEDDINGS-ALL-2015-to-2016-with-titles_20230331121228.pkl'
# args.pickle = '/var/www/html/medscape/medscape-content-feed/linear_model/storage/shared/df_cache/EMBEDDINGS-ALL-2023-with-titles/EMBEDDINGS-ALL-2023-with-titles_20230421193710.pkl'

pickle_file = args.pickle
data = pd.read_pickle(pickle_file)
if len(data.columns) == 5:
    data.columns = ['article_id', 'chunk', 'text', 'content_vector', 'title']
else:
    data.columns = ['article_id', 'chunk', 'text', 'content_vector']
data = data[['article_id', 'chunk', 'content_vector']]
data.head()

# Connect to Redis
#-------------------------------------------------------------------------------
import redis
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField
)

# Connect to Redis
redis_client = redis.Redis(
    host=env('REDIS_HOST'),
    port=env('REDIS_PORT'),
    password=env('REDIS_PASSWORD')
)
print(redis_client.ping())

# Creating a Search Index in Redis
#-------------------------------------------------------------------------------
# Constants
VECTOR_DIM = len(data.content_vector[0]) # length of the vectors
VECTOR_NUMBER = len(data)                 # initial number of vectors
INDEX_NAME = "test4-medscape-embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)

def create_index():
    # Define RediSearch fields for each of the columns in the dataset
    # title = TextField(name="title")
    article_id = NumericField(name="article_id")
    # text = TextField(name="text")
    # title_embedding = VectorField("title_vector",
    #     "FLAT", {
    #         "TYPE": "FLOAT32",
    #         "DIM": VECTOR_DIM,
    #         "DISTANCE_METRIC": DISTANCE_METRIC,
    #         "INITIAL_CAP": VECTOR_NUMBER,
    #     }
    # )
    text_embedding = VectorField("content_vector",
        "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": DISTANCE_METRIC,
            "INITIAL_CAP": VECTOR_NUMBER,
        }
    )
    # fields = [title, article_id, text, text_embedding]
    fields = [article_id, text_embedding]

    # Check if index exists
    try:
        redis_client.ft(INDEX_NAME).info()
        print("Index already exists")
    except:
        # Create RediSearch Index
        redis_client.ft(INDEX_NAME).create_index(
            fields = fields,
            definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )

def create_hnsw_index():
    #-------------------------------------------------------------------------------
    # HNSW Index
    #-------------------------------------------------------------------------------
    """
    Up until now, we've been using the FLAT or "brute-force" index to run our queries. Redis also supports the HNSW index which is a fast, approximate index. The HNSW index is a graph-based index that uses a hierarchical navigable small world graph to store vectors. The HNSW index is a good choice for large datasets where you want to run approximate queries.

    HNSW will take longer to build and consume more memory for most cases than FLAT but will be faster to run queries on, especially for large datasets.

    The following cells will show how to create an HNSW index and run queries with it using the same data as before.
    """
    # re-define RediSearch vector fields to use HNSW index
    # title_embedding = VectorField("title_vector",
    #     "HNSW", {
    #         "TYPE": "FLOAT32",
    #         "DIM": VECTOR_DIM,
    #         "DISTANCE_METRIC": DISTANCE_METRIC,
    #         "INITIAL_CAP": VECTOR_NUMBER
    #     }
    # )
    article_id = NumericField(name="article_id")
    text_embedding = VectorField("content_vector",
        "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": DISTANCE_METRIC,
            "INITIAL_CAP": VECTOR_NUMBER
        }
    )
    # fields = [title, article_id, text, title_embedding, text_embedding]
    fields = [article_id, text_embedding]

    import time
    # Check if index exists
    HNSW_INDEX_NAME = INDEX_NAME+ "_HNSW"

    try:
        redis_client.ft(HNSW_INDEX_NAME).info()
        print("Index already exists")
    except:
        # Create RediSearch Index
        redis_client.ft(HNSW_INDEX_NAME).create_index(
            fields = fields,
            definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
        )


#-------------------------------------------------------------------------------
# Load Documents into the Index
#-------------------------------------------------------------------------------
# Now that we have a search index, we can load documents into it. We will use the same documents we used in the previous examples.
# In Redis, either the Hash or JSON (if using RedisJSON in addition to RediSearch) data types can be used to store documents. We 
# will use the HASH data type in this example. The below cells will show how to load documents into the index.
import math
def index_documents(client: redis.Redis, prefix: str, documents: pd.DataFrame):
    records = documents.to_dict("records")
    numRecords = len(records)
    print('total records: %d' % numRecords)
    limit = 1000
    pages = math.ceil(numRecords / limit)
    i = 0
    page = 1
    for doc in records:
        if i % limit == 0:
            print("doing page %d of %d (%d per page)" % (page, pages, limit))
            page += 1
        #
        key = f"{prefix}:{str(doc['article_id'])}:chunk_{str(doc['chunk'])}"
        #
        # title_embedding = np.array(doc["title_vector"], dtype=np.float32).tobytes()
        content_embedding = np.array(doc["content_vector"], dtype=np.float32).tobytes()
        #
        # replace list of floats with byte vectors
        # doc["title_vector"] = title_embedding
        doc["content_vector"] = content_embedding
        #
        client.hset(key, mapping = doc)
        i += 1

create_index()
# create_hnsw_index()
index_documents(redis_client, PREFIX, data)
print(f"Loaded {redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")

print('done!')