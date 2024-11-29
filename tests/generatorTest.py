from generator import generator
import boto3

# Initialize AWS Bedrock client
bedrock_client = boto3.client('bedrock', region_name='us-east-1')

# Example query and results
query = "What are the key findings from the 2024 climate study?"
rr_results = [
    {
        "text": "The 2024 study highlights significant trends in global warming.",
        "metadata": {
            "similarity_score": 0.95,
            "recording_date": "2024-06-15",
            "contributors": ["Dr. Smith", "Dr. Lee"],
            "location": "NASA Climate Research"
        }
    },
    # Add more results as needed
]

# Initialize generator
gen = generator(query=query, bedrock_client=bedrock_client, rr_results=rr_results, rr_filenames=[], rr_best_filename="")

# Generate response
response, text_response = gen.generator(query, bedrock_client, rr_results)

print("Generated Response:", text_response)
