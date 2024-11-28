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
from generator import generator
from retriever import retriever
import streamlit as st

class HistOracleApp:
    def __init__(self, project_root):
        self.query = ""
        self.top_k = 5
        self.project_root = project_root
        self.vstore_name = ""

    def render_title(self):
        st.title("HistOracle")
        st.subheader("Library of Congress Research Assistant")

    def get_user_input(self):
        self.query = st.text_input("Enter your query:", "")
        self.top_k = st.sidebar.selectbox("Select the top-k documents to process:", options=[1, 3, 5, 8, 10])
        self.vstore_name = st.sidebar.selectbox("Select which vector store you would like to query against:", options=['vectorstore_all_250_instruct', 'vectorstore_all_1000_instruct', 'vectorstore_all_250_titan'])

    def validate_input(self):
        if not self.query.strip():
            st.error("Please enter a query.")
            return False
        elif not self.vstore_name.strip():
            st.error("Please select a vector store.")
            return False
        elif not self.top_k:
            st.error("Please select a top-k value.")
            return False
        return True

    def generate_response(self):
        try:
            # Run retriever
            st.write('Searching...')
            retriever_instance = retriever(project_root=self.project_root)  # Create the retriever instance
            query, bedrock_client, rr_results, rr_filenames, rr_best_filename = retriever_instance.runRetriever(
                query=self.query, top_k=self.top_k, vstore_name=self.vstore_name)
            st.write('Documents Found...')

            # Initialize the generator with the retrieved docs
            st.write('Generating Response...')
            generator_instance = generator(
                query=query,
                bedrock_client=bedrock_client,
                rr_results=rr_results,
                rr_filenames=rr_filenames,
                rr_best_filename=rr_best_filename,
            )

            # Generate the response
            response, text_response = generator_instance.generator(query, bedrock_client, rr_results)

            # Display results
            st.subheader("Text Response:")
            st.write(text_response)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    def run(self):
        self.render_title()
        self.get_user_input()
        if st.button("Generate Response") and self.validate_input():
            self.generate_response()
