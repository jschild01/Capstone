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
from langchain_community.vectorstores import DeepLake
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import json
import streamlit as st
from configparser import ConfigParser, ExtendedInterpolation
from langchain_aws import BedrockEmbeddings
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src/components"))
from src.components.generator import generator
from src.components.retriever import retriever
from src.components.create_vectorstore import AWSCredentialManager, RAGRetriever
from src.components.metadata_processor import process_metadata
from src.components.logging_config import setup_logging


class HistOracleApp:
    def __init__(self, project_root):
        self.query = ""
        self.top_k = 5
        self.project_root = project_root
        self.vstore_name = ""

    def render_title(self):
        st.title("FolkRAG")
        st.subheader("Library of Congress Research Assistant")

    def get_user_input(self):
        self.query = st.text_input("Enter your query:", "")
        self.top_k = st.sidebar.selectbox("Select the top-k documents to process:", options=[1, 3, 5, 8, 10])
        self.vstore_name = st.sidebar.selectbox(
            "Select which vector store you would like to query against:",
            options=[
                'vectorstore_all_250_instruct',
                'vectorstore_all_1000_instruct',
                'vectorstore_all_250_titan',
                'vectorstore_sample_250_instruct',
                'vectorstore_sample_1000_instruct'
            ]
        )

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

    def render_vector_store_creator(self):
        st.subheader("Create New Vector Store")

        # Add data structure information
        st.write("#### Required Data Structure")
        st.write("Your data directory must follow this structure:")

        directory_structure = """
        data/
        ├── txt/              # Plain text documents
        ├── transcripts/      # Text transcriptions
        ├── pdf/
        │   └── txtConversion/  # OCR-converted PDF text
        └── loc_dot_gov_data/   # Metadata directory
            └── {collection_name}/
                ├── file_list.csv
                └── search_results.csv
        """

        st.code(directory_structure, language="")

        st.write("#### Important Notes")
        st.markdown("""
        - All text files must be UTF-8 encoded
        - File names should follow AFC identifier pattern (e.g., afc2021007_002_ms01.txt)
        - Transcripts should be named as {original_name}_en.txt or {original_name}_en_translation.txt
        - Each collection needs file_list.csv and search_results.csv with proper metadata
        """)

        # Create vector store form
        st.write("#### Vector Store Creation")
        data_dir = st.text_input("Data Directory:", self.project_root)
        dataset_path = st.text_input("Dataset Output Path:", os.path.join(data_dir, 'vectorstore'))

        model_type = st.selectbox(
            "Embedding Model:",
            ["instructor", "mini", "titan"],
            format_func=lambda x: {
                "instructor": "Instructor-XL (Best quality, slower)",
                "mini": "MiniLM (Faster, lighter)",
                "titan": "Titan (Requires AWS credentials)"
            }[x]
        )

        batch_size = st.number_input("Batch Size:", value=1000, min_value=1)
        delete_existing = st.checkbox("Delete existing dataset if present")

        if st.button("Create Vector Store"):
            try:
                retriever = RAGRetriever(
                    dataset_path=dataset_path,
                    model_name=model_type
                )

                with st.spinner("Processing metadata..."):
                    metadata = process_metadata(data_dir)

                with st.spinner("Initializing vector store..."):
                    retriever.initialize_vectorstore(delete_existing=delete_existing)

                with st.spinner("Loading documents..."):
                    documents = retriever.load_data(data_dir, metadata)

                if documents:
                    with st.spinner("Generating embeddings..."):
                        retriever.process_with_checkpoints(documents, batch_size=batch_size)
                    st.success("Vector store created successfully!")
                else:
                    st.error("No documents found to process.")

            except Exception as e:
                st.error(f"Error creating vector store: {str(e)}")

    def render_query_interface(self):
        self.get_user_input()
        if st.button("Generate Response") and self.validate_input():
            self.generate_response()

    def generate_response(self):
        try:
            st.write('Searching...')
            retriever_instance = retriever(project_root=self.project_root)
            query, bedrock_client, rr_results, rr_filenames, rr_best_filename = retriever_instance.runRetriever(
                query=self.query, top_k=self.top_k, vstore_name=self.vstore_name)
            st.write('Documents Found...')

            st.write('Generating Response...')
            generator_instance = generator(
                query=query,
                bedrock_client=bedrock_client,
                rr_results=rr_results,
                rr_filenames=rr_filenames,
                rr_best_filename=rr_best_filename,
            )

            response, text_response = generator_instance.generator(query, bedrock_client, rr_results)

            st.subheader("Text Response:")
            st.write(text_response)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    def run(self):
        self.render_title()

        # Create tabs
        query_tab, vector_store_tab = st.tabs(["Query Interface", "Vector Store Creator"])

        # Render content for each tab
        with query_tab:
            self.render_query_interface()

        with vector_store_tab:
            self.render_vector_store_creator()