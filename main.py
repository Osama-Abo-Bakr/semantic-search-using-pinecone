# Import Library
from fastapi import FastAPI, Form, HTTPException
from utils import search_vectDB, delete_vectorDB, insert_vectorDB



# Initialize FastAPI
app = FastAPI(debug=True)

## ---- First Endpoint for searching the pinecone vectorDB ----- ##
# @app.post('/semantic_search')
# async def search(query_text: str=Form(...),
#                  top_k: int=Form(...),
#                  threshold: float=Form(None),
#                  class_type: str=Form(..., description='Class_Type', enum=['All', 'class-a', 'class-b'])):
    
#     # Some Validation.
#     if top_k <= 0 and not isinstance(top_k, int) or top_k >= 1000 or top_k is None:
#         raise HTTPException(status_code=400, detail='top_k must be a positive integer less than 1000')
    
#     elif threshold is not None and (threshold <= 0 or threshold > 1 or not isinstance(threshold, float)):
#         raise HTTPException(status_code=400, detail='threshold must be a float between 0 and 1')
    
#     elif class_type not in ['All', 'class-a', 'class-b']:
#         raise HTTPException(status_code=400, detail='class_type must be one of the following: All, class-a, class-b')
    
#     else:
#         # Search Vector DB
#         similar_ids = search_vectDB(query_text=query_text,
#                                     top_k=top_k,
#                                     threshold=threshold,
#                                     class_type=class_type)
    
#         return similar_ids


@app.post('/semantic_search')
async def semantic_search(search_text: str=Form(...), top_k: int=Form(100), 
                          threshold: float=Form(None), class_type: str=Form(..., description='class_type', enum=['All', 'class-a', 'class-b'])):

    ## Validation for top_k, and threshold
    if top_k <= 0 or not isinstance(top_k, int) or top_k > 10000 or top_k is None:
        raise HTTPException(status_code=400, detail="Bad Request: 'top_k' must be a positive integer and less than 10000.")
    
    elif threshold is not None and (threshold <= 0.0 or not isinstance(threshold, float) or threshold > 1.0):
        raise HTTPException(status_code=400, 
                            detail="Bad Request: 'threshold' must be a positive float greater than 0.0 and less than 1.0")

    else:
        ## Get Similar Records --> Call the (search_vectDB) from utils.py
        similar_records = search_vectDB(query_text=search_text, top_k=top_k, threshold=threshold, class_type=class_type)

        return similar_records
    

## ---- Second Endpoint for (Updating, Inserting) the pinecone vectorDB ----- ##
@app.post('/updating_or_deleting')
async def upserting_or_deleting(text_id: int=Form(...), text: str=Form(None),
                                class_type: str=Form(None, description='Class_Type', enum=['class-a', 'class-b']),
                                case: str=Form(..., description='Case', enum=['Upsert', 'Delete'])):
    
    
    if case == 'Upsert' and (not text or not class_type):
        raise HTTPException(status_code=400, detail='text and class_type are required for Upsert')
    
    # Call Function (inserting_vector) from utils
    if case == 'Upsert':
        msg = insert_vectorDB(text_id=text_id, text=text, class_type=class_type)
    
    # Call Function (deleting_vector) from utils
    elif case == 'Delete':
        msg = delete_vectorDB(ids=text_id)
    
    return {'message': msg}
