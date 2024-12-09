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
        st.write("### Required Data Structure")
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

        st.write("### Important Notes")
        st.markdown("""
        - All text files must be UTF-8 encoded
        - File names should follow AFC identifier pattern (e.g., afc2021007_002_ms01.txt)
        - Transcripts should be named as {original_name}_en.txt or {original_name}_en_translation.txt
        - Each collection needs file_list.csv and search_results.csv with proper metadata
        """)

        # Create vector store form
        st.write("### Vector Store Creation")
        data_dir = st.text_input("Data Directory:", self.project_root)
        dataset_path = st.text_input("Dataset Output Path:", os.path.join(data_dir, 'vectorstore'))

        chunk_size = st.select_slider(
            "Chunk Size (in characters):",
            options=[250, 500, 750, 1000, 1500, 2000],
            value=500,
            help="Controls how documents are split. Smaller chunks are better for finding specific information, larger chunks provide more context."
        )

        model_type = st.selectbox(
            "Embedding Model:",
            ["instructor", "mini", "titan"],
            format_func=lambda x: {
                "instructor": "Instructor-XL (Best quality, slower)",
                "mini": "MiniLM (Faster, lighter)",
                "titan": "Titan (Requires AWS credentials)"
            }[x]
        )

        batch_size = st.number_input(
            "Processing Batch Size:",
            value=1000,
            min_value=1,
            help="Number of chunks to process at once. Reduce if you encounter memory issues."
        )

        delete_existing = st.checkbox("Delete existing dataset if present")

        if st.button("Create Vector Store"):
            try:
                retriever = RAGRetriever(
                    dataset_path=dataset_path,
                    model_name=model_type,
                    chunk_size=chunk_size
                )

                with st.spinner("Processing metadata..."):
                    metadata = process_metadata(data_dir)

                    # Display sample metadata
                    st.write("### Sample Document Metadata")
                    if metadata:
                        # Get first 3 metadata entries
                        sample_entries = list(metadata.items())[:3]

                        for filename, meta in sample_entries:
                            with st.expander(f"📄 {filename}"):
                                # Basic Identification
                                st.markdown("#### 📑 Basic Identification")
                                st.write(f"• Call Number: {meta.get('call_number', 'N/A')}")
                                st.write(f"• Title: {meta.get('title', 'N/A')}")
                                st.write(f"• Type: {meta.get('type', 'N/A')}")

                                # Dates and Creation Info
                                st.markdown("#### 📅 Dates and Creation")
                                st.write(f"• Date: {meta.get('date', 'N/A')}")
                                st.write(f"• Created/Published: {meta.get('created_published', 'N/A')}")
                                st.write(f"• Timestamp: {meta.get('timestamp', 'N/A')}")

                                # Contributors and Sources
                                st.markdown("#### 👥 Contributors and Sources")
                                st.write(f"• Contributors: {meta.get('contributors', 'N/A')}")
                                st.write(f"• Creator: {meta.get('creator', 'N/A')}")
                                st.write(f"• Repository: {meta.get('repository', 'N/A')}")
                                st.write(f"• Collection: {meta.get('collection', 'N/A')}")
                                st.write(f"• Source Collection: {meta.get('source_collection', 'N/A')}")

                                # Content Details
                                st.markdown("#### 📝 Content Information")
                                st.write(f"• Language: {meta.get('language', 'N/A')}")
                                st.write(f"• Original Format: {meta.get('original_format', 'N/A')}")
                                st.write(f"• Online Formats: {meta.get('online_formats', 'N/A')}")
                                st.write(f"• Subjects: {meta.get('subjects', 'N/A')}")
                                st.write(f"• Locations: {meta.get('locations', 'N/A')}")

                                if meta.get('description'):
                                    st.markdown("#### 📋 Description")
                                    st.write(meta.get('description'))

                                if meta.get('notes'):
                                    st.markdown("#### 📒 Notes")
                                    st.write(meta.get('notes'))

                                # Collection Details
                                st.markdown("#### 📚 Collection Details")
                                st.write(f"• Collection Title: {meta.get('collection_title', 'N/A')}")
                                st.write(f"• Collection Date: {meta.get('collection_date', 'N/A')}")
                                if meta.get('collection_abstract'):
                                    st.write("**Collection Abstract:**")
                                    st.write(meta.get('collection_abstract'))
                                st.write(f"• Series Title: {meta.get('series_title', 'N/A')}")

                                # Catalog Information
                                st.markdown("#### 📗 Catalog Information")
                                st.write(f"• Catalog Title: {meta.get('catalog_title', 'N/A')}")
                                st.write(f"• Catalog Creator: {meta.get('catalog_creator', 'N/A')}")
                                st.write(f"• Catalog Date: {meta.get('catalog_date', 'N/A')}")
                                st.write(f"• Catalog Language: {meta.get('catalog_language', 'N/A')}")
                                st.write(f"• Catalog Genre: {meta.get('catalog_genre', 'N/A')}")
                                st.write(f"• Catalog Contributors: {meta.get('catalog_contributors', 'N/A')}")
                                st.write(f"• Catalog Repository: {meta.get('catalog_repository', 'N/A')}")
                                st.write(f"• Catalog Collection ID: {meta.get('catalog_collection_id', 'N/A')}")
                                if meta.get('catalog_description'):
                                    st.write("**Catalog Description:**")
                                    st.write(meta.get('catalog_description'))
                                if meta.get('catalog_subjects'):
                                    st.write("**Catalog Subjects:**")
                                    st.write(meta.get('catalog_subjects'))
                                if meta.get('catalog_notes'):
                                    st.write("**Catalog Notes:**")
                                    st.write(meta.get('catalog_notes'))

                                # Rights and Access
                                st.markdown("#### 🔒 Rights and Access")
                                st.write(f"• Rights: {meta.get('rights', 'N/A')}")
                                st.write(f"• Access Restricted: {meta.get('access_restricted', 'N/A')}")
                                st.write(f"• URL: {meta.get('url', 'N/A')}")

                        st.write(f"\nTotal documents with metadata: {len(metadata)}")
                    else:
                        st.warning("No metadata found in the specified directory.")

                with st.spinner("Initializing vector store..."):
                    retriever.initialize_vectorstore(delete_existing=delete_existing)

                with st.spinner("Loading documents..."):
                    documents = retriever.load_data(data_dir, metadata)

                if documents:
                    with st.spinner(
                            f"Processing documents into {chunk_size}-character chunks and generating embeddings..."):
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