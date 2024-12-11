import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException

_ = load_dotenv(override=True)

# Prepare Pinecone Index
PINCONE_API_KEY = os.getenv('PINECONE_API_KEY')

pinecone = Pinecone(api_key=PINCONE_API_KEY)
index_name = 'semantic-search-course'
index = pinecone.Index(index_name)


# Prepare Embedding Model
model_name = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name_or_path=model_name, device='cpu')

# ------------------------------ Translation using Groq ------------------------------
def translate_groq(text: str):
    ''' This Function takes the input text and translate it to English using gpt-3.5-turbo.

        Args:
        *****
            (user_prompt: str) --> The input text that we want to translate to English.
        
        Returns:
        ********
            (translated_text: str) --> The translation of the input text to English Language.
    '''

    client = Groq()
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": f"""
                You will provided with the following information.\n 
                1. An arbitrary input text. The text is delimited with triple backticks. \n\n 
                Perform the following tasks:\n
                1. Translate the following English text to English.\n 
                2. Return only the translation. Do not provide any additional information in your response.\n
                3. Also, Do not require any additional information for doing your tasks.\n\n
                Input text: {text}"""
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
    )

    return completion.choices[0].message.content


# -------------------------------------- Getting Similarity Score --------------------------------------

## ------------------------------------- Getting Similar IDs using pinecone ------------------------------------ ##
def search_vectDB(query_text: str, top_k: int, threshold: float=None, class_type: str=None):
    ''' This Function is to use the pinecone index to make a query and retrieve similar records.
    Args:
    *****
        (query_text: str) --> The query text to get similar records to it.
        (top_k: int) --> The number required of similar records in descending order.
        (threshold: float) --> The threshold to filter the retrieved IDs based on it.
        (class_type: str) --> Which class to filter using it (class-a or class-b)
    
    Returns:
    *******
        (similar_ids: List) --> A List of IDs for similarity records.
    '''
    try:
        ## Call the above Function for translation for better results
        # query_translated = translate_to_english_gpt(user_prompt=query_text)

        ## Get Embeddings of the input query
        query_embedding = embedding_model.encode(query_text).tolist()

        if class_type in ['class-a', 'class-b']:
            ## Search in pinecone with filtering using class_type
            results = index.query(vector=[query_embedding], top_k=top_k, 
                                  filter={'class': class_type}, include_metadata=True)
            results = results['matches']
        else: 
            ## Search in pinecone without filtering
            results = index.query(vector=[query_embedding], top_k=top_k, include_metadata=True)
            results = results['matches']

        print(results)
        
        ## Filter the output if there is a threshold given
        if threshold: 
            ## Exatract IDs with scores
            # similar_records = [{'id': int(record['id']), 'score': float(record['score']), 'class': record['metadata']['class']} \
                            #    for record in results if float(record['score']) > threshold]
            
            similar_records = [{'id': int(record['id']), 'score': float(record['score'])} \
                               for record in results if float(record['score']) > threshold]
       
        ## No Filtering
        else:
            ## Exatract IDs with scores
            # similar_records = [{'id': int(record['id']), 'score': float(record['score']), 'class': record['metadata']['class']} for record in results]
            
            similar_records = [{'id': int(record['id']), 'score': float(record['score'])} for record in results]

        return similar_records
    
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to get similar records' + str(e))
    
    

# ------------------------------------- Upserting New Data ----------------------------------------------
def insert_vectorDB(text_id: int, text: str, class_type: str):
    try:
        # Get Embedding
        embeds = embedding_model.encode(text).tolist()
        
        # Upserting to Vector DB
        to_upsert = [(str(text_id), embeds, {'class': class_type})]
        
        # Upsert to Pinecone
        _ = index.upsert(to_upsert)
        
        # Get the count of vectorDB
        count_after = index.describe_index_stats()['total_vector_count']
        
        return f'Upserting Done: Count Now is {count_after} vectors.'
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to Upsert New Data')
    
    
# ------------------------------------- Deleting New Data ----------------------------------------------
def delete_vectorDB(ids: int):
    try:
        # Delete from Pinecone
        _ = index.delete(ids=[str(ids)])
        
        # Get the count of vectorDB
        count_after = index.describe_index_stats()['total_vector_count']
        
        return f'Deleting Done: Count Now is {count_after} vectors.'
    
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to Delete Data')