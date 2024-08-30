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
            """Advanced Data Acquisition and Web Mining""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The goal of this project is to develop a comprehensive, intelligent web scraping and data structuring system
             tailored for research applications. This system will:

            1. Implement a diverse range of web scraping techniques to collect data from various online sources relevant 
            to academic and industry research.
            2. Utilize machine learning and natural language processing to automatically identify, categorize, and 
            extract relevant information from scraped content.
            3. Develop a robust data cleaning and structuring pipeline to transform raw scraped data into organized, 
            analysis-ready datasets.
            4. Create an intelligent source discovery mechanism that can automatically find and suggest new relevant 
            data sources based on the research topic.
            5. Implement a user-friendly interface for researchers to configure scraping tasks, monitor progress, and 
            access structured data.
            6. Ensure ethical scraping practices, including respect for robots.txt, rate limiting, and compliance with 
            terms of service.
            7. Incorporate data validation and quality assurance measures to ensure the reliability of collected information.

            The system will leverage the following Python packages and tools:

            Web Scraping:
            - Requests: For making HTTP requests to websites
            - BeautifulSoup4: For parsing HTML and XML documents
            - Scrapy: For building comprehensive web crawlers
            - Selenium: For automating browser interactions with dynamic content
            - PyPDF2: For extracting text from PDF documents
            - python-docx: For handling Microsoft Word documents
            - Newspaper3k: For article scraping and curation
            - Trafilatura: For web scraping, crawling, and text extraction

            Data Cleaning and Structuring:
            - Pandas: For data manipulation and analysis
            - NumPy: For numerical operations on structured data
            - NLTK (Natural Language Toolkit): For text processing and linguistic data cleaning
            - spaCy: For advanced natural language processing tasks
            - scikit-learn: For data preprocessing and feature extraction
            - Fuzzywuzzy: For string matching and deduplication
            - dateparser: For parsing diverse date formats
            - PyJanitor: For data cleaning functions built on top of Pandas

            Machine Learning and NLP:
            - TensorFlow or PyTorch: For implementing custom ML models for data classification and extraction
            - Transformers: For utilizing pre-trained language models for text analysis
            - Gensim: For topic modeling and document similarity analysis
            - fastText: For efficient text classification

            Visualization and Reporting:
            - Matplotlib and Seaborn: For creating static visualizations
            - Plotly: For interactive data visualizations
            - Dash: For building analytical web applications

            Utility and Automation:
            - Airflow: For orchestrating and scheduling complex scraping workflows
            - Celery: For distributed task queue management
            - Beautiful Soup Sitemaps: For parsing XML sitemaps to discover web pages
            - tqdm: For adding progress bars to long-running tasks

            The end product will be a powerful, flexible system capable of autonomously collecting, cleaning, and 
            structuring large volumes of web data from diverse sources, tailored to support various research endeavors 
            across academic and industry settings.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            The project will target a diverse range of online sources relevant to research, including but not limited to:

            1. Academic journal websites and digital libraries (e.g., JSTOR, ScienceDirect)
            2. Preprint servers (e.g., arXiv, bioRxiv)
            3. Conference proceedings and presentation repositories
            4. Government databases and reports
            5. Industry publications and whitepapers
            6. News outlets and press releases related to research topics
            7. Social media platforms for tracking research discussions and trends
            8. Patent databases
            9. Open data repositories (e.g., data.gov, Kaggle datasets)
            10. Research institution websites

            Students will need to identify and prioritize data sources based on their relevance, accessibility, and data
             quality. They should also consider the specific requirements of different research domains when selecting 
             and structuring the data.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            In the era of information abundance, efficient data acquisition and structuring are crucial for advancing 
            research across various fields. This project addresses several key needs:

            1. Automation of time-consuming data collection processes, allowing researchers to focus on analysis and 
            interpretation
            2. Improvement of data quality and consistency through standardized cleaning and structuring processes
            3. Enhanced discovery of relevant information sources, potentially uncovering valuable but lesser-known data
            4. Democratization of access to web data, enabling smaller research teams to work with large-scale datasets
            5. Promotion of reproducibility in research by providing a systematic approach to data collection
            6. Development of transferable skills in web scraping, data processing, and machine learning that are valuable
             in both academic and industry settings
            7. Creation of a flexible tool that can adapt to the evolving landscape of online information sources

            By developing this advanced research data acquisition system, students will contribute to the efficiency and 
            effectiveness of the research process across multiple disciplines, potentially accelerating scientific 
            discovery and innovation.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            The project will be approached through several key steps:

            1. Research and Planning:
               - Study existing web scraping techniques and tools
               - Analyze challenges in current research data collection practices
               - Define system requirements and architecture

            2. Development of Core Scraping Modules:
               - Implement various scraping techniques (e.g., HTML parsing, API interactions, browser automation)
               - Develop modules for handling different data formats (e.g., HTML, PDF, DOC)
               - Create a unified interface for managing diverse scraping tasks

            3. Intelligent Source Discovery System:
               - Develop algorithms to analyze and categorize potential data sources
               - Implement machine learning models for relevance scoring of new sources
               - Create a recommendation system for suggesting new data sources to researchers

            4. Data Cleaning and Structuring Pipeline:
               - Develop robust data cleaning algorithms to handle inconsistencies and errors
               - Implement intelligent data parsing to extract structured information from unstructured content
               - Create flexible data models to accommodate various research data types

            5. Machine Learning and NLP Integration:
               - Implement text classification models for automatic content categorization
               - Develop named entity recognition systems for extracting key information
               - Create topic modeling algorithms for content summarization and organization

            6. User Interface and Visualization:
               - Design and implement a user-friendly interface for configuring scraping tasks
               - Develop real-time monitoring and reporting features
               - Create interactive data visualization tools for exploring scraped datasets

            7. Ethical and Legal Compliance System:
               - Implement robust rate limiting and respect for robots.txt
               - Develop a system for managing and respecting data usage rights
               - Create audit trails for data provenance

            8. Integration and Scalability:
               - Combine all modules into a cohesive system
               - Implement distributed computing capabilities for handling large-scale scraping tasks
               - Develop APIs for potential integration with other research tools

            9. Testing and Optimization:
               - Conduct thorough testing on various websites and data types
               - Optimize performance, focusing on speed, accuracy, and resource utilization
               - Implement iterative improvements based on user feedback

            10. Documentation and Deployment:
                - Create comprehensive documentation for the system
                - Prepare a deployment strategy, including considerations for cloud-based implementations
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            This is a rough timeline for the project:

            - (2 Weeks) Research and Planning
            - (4 Weeks) Development of Core Scraping Modules
            - (3 Weeks) Intelligent Source Discovery System
            - (3 Weeks) Data Cleaning and Structuring Pipeline
            - (3 Weeks) Machine Learning and NLP Integration
            - (3 Weeks) User Interface and Visualization
            - (2 Weeks) Ethical and Legal Compliance System
            - (3 Weeks) Integration and Scalability
            - (3 Weeks) Testing and Optimization
            - (2 Weeks) Documentation and Deployment
            - (2 Weeks) Final Presentation and Project Wrap-up
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            This project is suitable for a team of 1-2 students. The complexity and scope of the project allow for effective distribution of tasks among team members, promoting collaborative learning and development across various aspects of web scraping, data processing, and machine learning.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Several challenges may arise during the project:

            1. Ethical and Legal Considerations: Ensuring compliance with websites' terms of service, respecting robots.txt files, and adhering to data usage rights.
            2. Anti-Scraping Measures: Dealing with CAPTCHAs, IP blocking, and other protective measures implemented by websites.
            3. Data Quality and Consistency: Ensuring the scraped data is accurate, complete, and properly structured across different sources.
            4. Scalability: Optimizing the system to handle large-scale scraping tasks efficiently without overwhelming target websites.
            5. Flexibility vs. Specificity: Balancing the need for a general-purpose tool with the requirements of specific research domains.
            6. Dynamic Content Handling: Effectively scraping websites with heavy JavaScript usage and dynamic loading.
            7. Data Integration: Combining data from multiple sources with potentially different formats and structures.
            8. Privacy and Sensitivity: Handling potentially sensitive or personal information in compliance with relevant regulations.
            9. Keeping Up with Web Technologies: Adapting to new web technologies and scraping techniques as they emerge.
            10. User Experience: Creating an intuitive interface that caters to users with varying levels of technical expertise.
            11. Performance Optimization: Balancing the thoroughness of data collection and processing with system responsiveness.
            12. Automated Decision Making: Ensuring the accuracy and reliability of the intelligent source discovery and data categorization features.

            Students will need to research and implement solutions to these challenges, which will be an integral part of the learning experience and contribute significantly to the project's innovation in the field of research data acquisition and management.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Dr. Amir Jafari",
        "Proposed by email": "ajafari@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gmail.com",
        "github_repo": "https://github.com/amir-jafari/Capstone",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")
