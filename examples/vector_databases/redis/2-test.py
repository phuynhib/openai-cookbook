from app import env
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

data = pd.read_pickle('data.pkl')
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
    VectorField
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
VECTOR_DIM = len(data['title_vector'][0]) # length of the vectors
VECTOR_NUMBER = len(data)                 # initial number of vectors
INDEX_NAME = "embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)

# Define RediSearch fields for each of the columns in the dataset
title = TextField(name="title")
url = TextField(name="url")
text = TextField(name="text")
title_embedding = VectorField("title_vector",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    }
)
text_embedding = VectorField("content_vector",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    }
)
fields = [title, url, text, title_embedding, text_embedding]

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

# Load Documents into the Index
#-------------------------------------------------------------------------------
# Now that we have a search index, we can load documents into it. We will use the same documents we used in the previous examples.
# In Redis, either the Hash or JSON (if using RedisJSON in addition to RediSearch) data types can be used to store documents. We 
# will use the HASH data type in this example. The below cells will show how to load documents into the index.
def index_documents(client: redis.Redis, prefix: str, documents: pd.DataFrame):
    records = documents.to_dict("records")
    for doc in records:
        key = f"{prefix}:{str(doc['id'])}"
        #
        # create byte vectors for title and content
        title_embedding = np.array(doc["title_vector"], dtype=np.float32).tobytes()
        content_embedding = np.array(doc["content_vector"], dtype=np.float32).tobytes()
        #
        # replace list of floats with byte vectors
        doc["title_vector"] = title_embedding
        doc["content_vector"] = content_embedding
        #
        client.hset(key, mapping = doc)

index_documents(redis_client, PREFIX, data)
print(f"Loaded {redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")
"""
Loaded 25000 documents in Redis search index with name: embeddings-index
"""

# Simple Vector Search Queries with OpenAI Query Embeddings
#-------------------------------------------------------------------------------
# Now that we have a search index and documents loaded into it, we can run search queries.
# Below we will provide a function that will run a search query and return the results. 
# Using this function we run a few queries that will show how you can utilize Redis as a vector database.
def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = "embeddings-index",
    vector_field: str = "title_vector",
    return_fields: list = ["title", "url", "text", "vector_score"],
    hybrid_fields = "*",
    k: int = 20,
    print_results: bool = True,
) -> List[dict]:
    #
    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(input=user_query,
                                            model="text-embedding-ada-002",
                                            )["data"][0]['embedding']
    #
    # Prepare the Query
    base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
    query = (
        Query(base_query)
         .return_fields(*return_fields)
         .sort_by("vector_score")
         .paging(0, k)
         .dialect(2)
    )
    params_dict = {"vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()}
    #
    # perform vector search
    results = redis_client.ft(index_name).search(query, params_dict)
    if print_results:
        for i, article in enumerate(results.docs):
            score = 1 - float(article.vector_score)
            print(f"{i}. {article.title} (Score: {round(score ,3) })")
    return results.docs

# For using OpenAI to generate query embedding
results = search_redis(redis_client, 'modern art in Europe', k=10)
"""
0. Museum of Modern Art (Score: 0.875)
1. Western Europe (Score: 0.868)
2. Renaissance art (Score: 0.864)
3. Pop art (Score: 0.86)
4. Northern Europe (Score: 0.855)
5. Hellenistic art (Score: 0.853)
6. Modernist literature (Score: 0.847)
7. Art film (Score: 0.843)
8. Central Europe (Score: 0.843)
9. European (Score: 0.841)
"""

results = search_redis(redis_client, 'Famous battles in Scottish history', vector_field='content_vector', k=10)
"""
0. Battle of Bannockburn (Score: 0.869)
1. Wars of Scottish Independence (Score: 0.861)
2. 1651 (Score: 0.853)
3. First War of Scottish Independence (Score: 0.85)
4. Robert I of Scotland (Score: 0.846)
5. 841 (Score: 0.844)
6. 1716 (Score: 0.844)
7. 1314 (Score: 0.837)
8. 1263 (Score: 0.836)
9. William Wallace (Score: 0.835)
"""

# Hybrid Queries with Redis
#-------------------------------------------------------------------------------
# The previous examples showed how run vector search queries with RediSearch. In this section, we will show how to combine vector search with other RediSearch fields for hybrid search. In the below example, we will combine vector search with full text search.
def create_hybrid_field(field_name: str, value: str) -> str:
    return f'@{field_name}:"{value}"'

# search the content vector for articles about famous battles in Scottish history and only include results with Scottish in the title
results = search_redis(redis_client,
                       "Famous battles in Scottish history",
                       vector_field="title_vector",
                       k=5,
                       hybrid_fields=create_hybrid_field("title", "Scottish")
                       )
"""
0. First War of Scottish Independence (Score: 0.892)
1. Wars of Scottish Independence (Score: 0.889)
2. Second War of Scottish Independence (Score: 0.879)
3. List of Scottish monarchs (Score: 0.873)
4. Scottish Borders (Score: 0.863)
"""

# run a hybrid query for articles about Art in the title vector and only include results with the phrase "Leonardo da Vinci" in the text
results = search_redis(redis_client,
                       "Art",
                       vector_field="title_vector",
                       k=5,
                       hybrid_fields=create_hybrid_field("text", "Leonardo da Vinci")
                       )

# find specific mention of Leonardo da Vinci in the text that our full-text-search query returned
mention = [sentence for sentence in results[0].text.split("\n") if "Leonardo da Vinci" in sentence][0]
mention
"""
0. Art (Score: 1.0)
1. Paint (Score: 0.896)
2. Renaissance art (Score: 0.88)
3. Painting (Score: 0.874)
4. Renaissance (Score: 0.846)
'In Europe, after the Middle Ages, there was a "Renaissance" which means "rebirth". People rediscovered science and artists were allowed to paint subjects other than religious subjects. People like Michelangelo and Leonardo da Vinci still painted religious pictures, but they also now could paint mythological pictures too. These artists also invented perspective where things in the distance look smaller in the picture. This was new because in the Middle Ages people would paint all the figures close up and just overlapping each other. These artists used nudity regularly in their art.'
"""

# HNSW Index
#-------------------------------------------------------------------------------
"""
Up until now, we've been using the FLAT or "brute-force" index to run our queries. Redis also supports the HNSW index which is a fast, approximate index. The HNSW index is a graph-based index that uses a hierarchical navigable small world graph to store vectors. The HNSW index is a good choice for large datasets where you want to run approximate queries.

HNSW will take longer to build and consume more memory for most cases than FLAT but will be faster to run queries on, especially for large datasets.

The following cells will show how to create an HNSW index and run queries with it using the same data as before.
"""
# re-define RediSearch vector fields to use HNSW index
title_embedding = VectorField("title_vector",
    "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER
    }
)
text_embedding = VectorField("content_vector",
    "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER
    }
)
fields = [title, url, text, title_embedding, text_embedding]

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

# since RediSearch creates the index in the background for existing documents, we will wait until
# indexing is complete before running our queries. Although this is not necessary for the first query,
# some queries may take longer to run if the index is not fully built. In general, Redis will perform
# best when adding new documents to existing indices rather than new indices on existing documents.
max_checks=10
n = 1
while redis_client.ft(HNSW_INDEX_NAME).info()["indexing"] == "1":
    print('checking index progress %d of %d' % (n, max_checks))
    if n >= max_checks:
        break
    n += 1
    time.sleep(5)

results = search_redis(redis_client, 'modern art in Europe', index_name=HNSW_INDEX_NAME, k=10)
"""
0. Western Europe (Score: 0.868)
1. Northern Europe (Score: 0.855)
2. Central Europe (Score: 0.843)
3. European (Score: 0.841)
4. Eastern Europe (Score: 0.839)
5. Europe (Score: 0.839)
6. Western European Union (Score: 0.837)
7. Southern Europe (Score: 0.831)
8. Western civilization (Score: 0.83)
9. Council of Europe (Score: 0.827)
"""

# compare the results of the HNSW index to the FLAT index and time both queries
def time_queries(iterations: int = 10):
    print(" ----- Flat Index ----- ")
    t0 = time.time()
    for i in range(iterations):
        results_flat = search_redis(redis_client, 'modern art in Europe', k=10, print_results=False)
    t0 = (time.time() - t0) / iterations
    results_flat = search_redis(redis_client, 'modern art in Europe', k=10, print_results=True)
    print(f"Flat index query time: {round(t0, 3)} seconds\n")
    time.sleep(1)
    print(" ----- HNSW Index ------ ")
    t1 = time.time()
    for i in range(iterations):
        results_hnsw = search_redis(redis_client, 'modern art in Europe', index_name=HNSW_INDEX_NAME, k=10, print_results=False)
    t1 = (time.time() - t1) / iterations
    results_hnsw = search_redis(redis_client, 'modern art in Europe', index_name=HNSW_INDEX_NAME, k=10, print_results=True)
    print(f"HNSW index query time: {round(t1, 3)} seconds")
    print(" ------------------------ ")
time_queries()
"""
 ----- Flat Index ----- 
0. Museum of Modern Art (Score: 0.875)
1. Western Europe (Score: 0.867)
2. Renaissance art (Score: 0.864)
3. Pop art (Score: 0.861)
4. Northern Europe (Score: 0.855)
5. Hellenistic art (Score: 0.853)
6. Modernist literature (Score: 0.847)
7. Art film (Score: 0.843)
8. Central Europe (Score: 0.843)
9. Art (Score: 0.842)
Flat index query time: 0.263 seconds

 ----- HNSW Index ------ 
0. Western Europe (Score: 0.867)
1. Northern Europe (Score: 0.855)
2. Central Europe (Score: 0.843)
3. European (Score: 0.841)
4. Eastern Europe (Score: 0.839)
5. Europe (Score: 0.839)
6. Western European Union (Score: 0.837)
7. Southern Europe (Score: 0.831)
8. Western civilization (Score: 0.83)
9. Council of Europe (Score: 0.827)
HNSW index query time: 0.129 seconds
 ------------------------ 
"""