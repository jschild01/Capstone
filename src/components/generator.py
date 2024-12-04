import os
import datetime
import re
import numpy as np
import pandas as pd
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from typing import Dict, List, Any
from langchain.vectorstores import DeepLake
from langchain_community.vectorstores import DeepLake # last one was depracated
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import json
import streamlit as st
from configparser import ConfigParser, ExtendedInterpolation
from langchain_aws import BedrockEmbeddings

class generator:
    def __init__(self, query, bedrock_client, rr_results, rr_filenames, rr_best_filename):
        self.query = query
        self.bedrock_client = bedrock_client
        self.rr_results = rr_results
        self.rr_filenames = rr_filenames
        self.rr_best_filename = rr_best_filename

    # all the below need for generator
    def generate_prompt_claude(self, query, context, metadata):
        # Formatting metadata into a string, each key-value pair on a new line
        if isinstance(metadata, list) and all(isinstance(item, dict) for item in metadata):
            # Formatting metadata into a string, each key-value pair on a new line
            metadata_str = "\n".join([f"{k}: {v}" for item in metadata for k, v in item.items()])
        elif isinstance(metadata, dict):
            # If metadata is a single dictionary, handle it directly
            metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
        elif isinstance(metadata, str):
            # If metadata is a string, use it as-is
            metadata_str = metadata
        else:
            # If metadata is an unexpected type, handle gracefully
            metadata_str = "Invalid metadata format"
        
        # Constructing the prompt with the required prefix and structured information
        return f"""Human: Please answer the following query based on the provided context and metadata.
    Query: {query}
    Context: {context}
    Metadata: {metadata_str}

    Instructions: 
    1. Answer the question using ONLY the information provided in the Context and Metadata above.
    2. Do NOT include any information that is not explicitly stated in the Context or Metadata.
    3. Begin your answer with a direct response to the question asked.
    4. Include relevant details from the Context and Metadata to support your answer.
    5. Pay special attention to the recording date, contributors, and locations provided in the metadata.
    6. Inform the user of what document filename they can find the information in.

    Your Answer here:"""

    def generate_response(self, bedrock_client, prompt, max_length=300):
        request_payload = json.dumps({"anthropic_version": 
                                    "bedrock-2023-05-31", 
                                    "messages": [
                                                    {
                                                        "role": "user",
                                                        "content": prompt
                                                    }
                                                ],
                                    "max_tokens": max_length,  # Corrected key from max_tokens_to_sample to max_tokens
                                    "temperature": 0.7
                                    }).encode('utf-8')

        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=request_payload,
            contentType='application/json'
        )

        # Decode the response assuming it returns JSON with 'generated_text' key
        response_body = response['body'].read().decode()
        return response_body

    def generator(self, query, bedrock_client, rr_results):

        # extract text from rr_results to feed to the generator (will have text for each doc in top_ks)
        texts = [result['text'] for result in rr_results]

        # extract metadata from rr_results to feed to the generator (will have text for each doc in top_ks)
        # would we really want to feed the LLM the metadata for, say, topK of 20????? just get the best doc's metadata...
        #metadatas = [result['metadata'] for result in rr_results]

        # get the metadata for the highest similarity scoring doc
        best_metadata = max(rr_results, key=lambda x: x['metadata']['similarity_score'])['metadata']

        # generate prompt for LLM
        llm_prompt = self.generate_prompt_claude(query, texts, best_metadata)

        # get LLM response
        response = self.generate_response(bedrock_client, llm_prompt, max_length=300)

        # get just the raw text response from the LLM
        response_dict = json.loads(response)
        text_response = response_dict["content"][0]["text"]

        return response, text_response



