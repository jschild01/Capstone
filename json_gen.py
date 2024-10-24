import json
import os
import shutil


def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)


data_to_save = \
    {
        # -----------------------------------------------------------------------------------------------------------------------
        "Version":
            """1""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2024""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Fall""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Natural Language Query Interface for Library Data""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The goal of this project is to develop an advanced natural language query interface for Library of Congress (LOC) 
            data, combining aspects of web scraping, data structuring, and multi-agent Large Language Models (LLMs) with 
            Retrieval-Augmented Generation (RAG).

            The system will:

            1. Implement a comprehensive web scraping and data acquisition pipeline to collect diverse data types from the 
            Library of Congress, including XML documents, catalog records, audio/video content, PDFs, and datasets.
            2. Develop robust data extraction and transcription processes for audio, video, and PDF content.
            3. Create a robust data cleaning and structuring pipeline to transform raw scraped data into organized, 
            analysis-ready datasets.
            4. Create a unified database or index of the collected and structured data.
            5. Implement a multi-agent LLM system using RAG techniques to enable natural language querying of the collected data.
            6. Develop a user-friendly interface for submitting queries and displaying results, including source citations.
            7. Ensure ethical data collection practices, including respect for rate limits and terms of service.
            8. Incorporate data validation and quality assurance measures to ensure the reliability of collected information.
            
            Key Components and Technologies (specific libraries are subject to change):
            
            Web Scraping and Data Acquisition:
            - Utilize libraries such as Requests, BeautifulSoup4, and Scrapy for web scraping
            - Implement API interactions for rate-limited resources
            
            Audio and Video Transcription:
            - Implement Whisper for transcribing audio content
            - Utilize video processing libraries (FFMPEG) for extracting audio from video files
            
            PDF Scraping:
            - Use PyPDF2 or pdf2image for PDF parsing and text extraction
            - Implement OCR (Optical Character Recognition) using libraries like Tesseract for scanned PDFs with no 
            embedded OCR.
            - Develop methods for preserving document structure and layout information
            
            Data Cleaning and Structuring:
            - Use Pandas and NumPy for data manipulation and cleaning
            - Implement natural language processing tools like NLTK or spaCy for text processing
            - Develop custom algorithms for handling domain-specific data formats
            
            Database and Indexing:
            - Utilize vector databases like Faiss or Pinecone for efficient similarity search
            - Implement document indexing for quick retrieval
            - Design a flexible schema to accommodate diverse data types
            
            Multi-Agent LLM System:
            - Integrate LLMs such as GPT-4 or LLAMA
            - Implement a RAG system using frameworks like LangChain or Hugging Face's RAG model
            - Develop specialized agents for different types of library data (e.g., catalog records, archival
             descriptions, transcriptions, PDF content)
             
             User Interface:
             - Create a web-based interface using Streamlit
             
             Evaluation and Optimization:
             - Develop comprehensive evaluation metrics for query relevance and accuracy
             - Implement mechanisms for continuous system improvement based on user feedback
             

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            The project will focus on data from the Library of Congress, including but not limited to:

            1. Collection guide (finding aid) XML: From https://findingaids.loc.gov/exist_collections/ead3master/
            2. Catalog record XML: From https://lccn.loc.gov/[record_number]/marcxml
            3. Audio, video, textual, and PDF content: Accessible via API from https://www.loc.gov/collections/
            4. Datasets: From https://www.loc.gov/collections/selected-datasets/
            5. Web archives: From https://labs.loc.gov/work/experiments/webarchive-datasets/
            6. Additional datasets: 
            From https://catalog.data.gov/dataset/?q=library+of+congress&_organization_limit=0&organization=library-of-congress

            The project will involve creating a structured database from these diverse sources, which will serve as a 
            contribution to the open-source community. The database will be shared in a structured format for reproducibility 
            and future use by other researchers.

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            This project addresses several key needs in the field of library science and information retrieval:
            
            1. Unified Access: Enables users to query diverse library resources through a single, intuitive interface.
            2. Natural Language Understanding: Allows users to ask complex questions without needing to understand specific query 
            languages or database structures.
            3. Cross-Resource Integration: Facilitates the discovery of connections between different types of library resources.
            4. Improved Accessibility: Makes vast amounts of library data more accessible to researchers and the general public.
            5. Citation Tracking: Provides accurate source information, crucial for academic and research purposes.
            6. Scalability: Creates a framework that could potentially be expanded to include data from multiple institutions.
            7. Open Data Contribution: Produces a structured, open-source database of library data for future research and 
            development.

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            The project will be approached through several key steps:

            1. Data Acquisition and Preprocessing:
               - Develop web scraping scripts for XML and catalog data
               - Implement API interaction for audio, video, text, and PDF content
               - Create a pipeline for downloading and organizing datasets and web archives
               - Develop audio and video transcription systems
               
            2. Data Cleaning and Structuring:
               - Implement data cleaning algorithms to handle inconsistencies across different data types
               - Develop a unified data model to represent diverse library resources
               - Create a pipeline for structuring and indexing the cleaned data
            
            3. Database and Index Creation:
               - Set up a vector database for efficient similarity search
               - Implement document indexing for quick information retrieval
               - Develop a system for regular updates and maintenance of the database
            
            4. Multi-Agent LLM System Development:
               - Integrate and fine-tune LLMs for library-specific tasks
               - Implement RAG techniques to enhance query responses with relevant context
               - Develop specialized agents for different types of library data
            
            5. Natural Language Query Processing:
               - Create a query understanding module to interpret user questions
               - Develop a query routing system to direct questions to appropriate agents
               - Implement a response synthesis module to combine information from multiple sources
            
            6. User Interface Development:
               - Design and implement a web-based frontend for query input and result display
               - Create visualizations for displaying relationships between different data sources
               - Implement features for result filtering and sorting
               
            7. Citation and Source Tracking:
               - Develop a system for tracking and displaying the sources of information in responses
               - Implement proper citation generation for academic use
            
            8. Evaluation and Optimization:
               - Create a set of test queries covering various aspects of library data
               - Implement evaluation metrics for response relevance, accuracy, and completeness
               - Develop a feedback mechanism for continuous system improvement
            
            9. Documentation and Deployment:
               - Create comprehensive documentation for the system architecture and usage
               - Prepare the structured database for open-source release
               - Develop a deployment strategy, including considerations for scaling and maintenance

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            This is a rough timeline for the project:

            - (2 Weeks) Data Acquisition and Preprocessing
            - (2 Weeks) Data Cleaning and Structuring
            - (2 Weeks) Database and Index Creation
            - (4 Weeks) Multi-Agent LLM System Development
            - (3 Weeks) Natural Language Query Processing
            - (2 Weeks) User Interface Development
            - (2 Weeks) Citation and Source Tracking
            - (2 Weeks) Documentation and Deployment
            - (1 Week) Final Presentation and Project Wrap-up
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            This project will be undertaken by 2 students. The complexity and scope of the project allow for effective 
            distribution of tasks between team members, promoting collaborative learning and development across various aspects 
            of data science, natural language processing, and information retrieval.

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Potential challenges include:

            1. Data Volume and Diversity: Managing and integrating large volumes of diverse data types from different Library of 
            Congress sources.
            2. API Rate Limits: Developing strategies to work within the constraints of API rate limits for certain data sources.
            3. Audio and Video Transcription Accuracy: Ensuring high-quality transcriptions of audio and video content, especially 
            for historical or low-quality recordings.
            4. PDF Extraction Challenges: Dealing with complex layouts, scanned documents, and preserving structural information 
            in PDFs.
            5. Data Consistency: Maintaining consistency in data structure and quality across different sources and formats.
            6. Query Interpretation: Accurately interpreting complex, ambiguous, or domain-specific natural language queries.
            7. Response Accuracy: Ensuring the accuracy and relevance of responses, especially when combining information from 
            multiple sources.
            8. Performance Optimization: Balancing system responsiveness with the depth and breadth of data searches.
            9. Scalability: Designing the system to handle potential future expansion to include data from other institutions.
            10. User Experience: Creating an intuitive interface that caters to users with varying levels of research experience.
            11. Ethical Considerations: Ensuring proper handling of potentially sensitive or copyrighted information in the 
            Library of Congress data.
            12. Citation Accuracy: Developing a robust system for tracking and citing sources accurately across diverse data types.
            13. Evaluation Metrics: Defining comprehensive metrics to assess the system's performance in handling diverse 
            library-related queries.
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Paul Kelly and Jonathan Schild",
        "Proposed by email": "pjameskelly@gwu.edu, jschild01@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gmail.com",
        "github_repo": "https://github.com/jschild01/Capstone",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")
