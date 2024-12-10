import re




def generate_prompt_claude(query: str, context: str, metadata: dict) -> str:
    # Formatting metadata into a string, each key-value pair on a new line
    metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    
    # Constructing the prompt with the required prefix and structured information
    return f"""Human: Please answer the following query based on the provided context and metadata.
Question: {query}
Context: {context}
Metadata: {metadata_str}

Instructions: 
1. Answer the question using ONLY the information provided in the Context and Metadata above.
2. Do NOT include any information that is not explicitly stated in the Context or Metadata.
3. If the information provided is not sufficient to answer the question fully, state this clearly.
4. Begin your answer with a direct response to the question asked.
5. Include relevant details from the Context and Metadata to support your answer.
6. Pay special attention to the recording date, contributors, and locations provided in the metadata.

Assistant: Answer:"""

def generate_prompt(query: str, context: str, metadata: dict) -> str:
    metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    return f"""Question: {query}
Context: {context}
Metadata: {metadata_str}

Instructions: 
1. Answer the question using ONLY the information provided in the Context and Metadata above.
2. Do NOT include any information that is not explicitly stated in the Context or Metadata.
3. If the information provided is not sufficient to answer the question fully, state this clearly.
4. Begin your answer with a direct response to the question asked.
5. Include relevant details from the Context and Metadata to support your answer.
6. Pay special attention to the recording date, contributors, and locations provided in the metadata.

Answer:"""

def generate_strict_prompt(query: str, context: str, metadata: dict) -> str:
    metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    return f"""STRICT INSTRUCTIONS: Adhere to these rules precisely when answering the question.

Question: {query}

Context:
{context}

Metadata:
{metadata_str}

Rules:
1. Use ONLY the information provided in the Context and Metadata above to answer the question.
2. Do NOT include any information that is not explicitly stated in the Context or Metadata.
3. Do NOT make assumptions, speculate, or use prior knowledge not present in the given information.
4. If the information to answer the question is not available, explicitly state: "The provided information does not contain an answer to this question."
5. Begin your answer with a direct response to the question asked.
6. After the direct response, provide supporting details from the Context and Metadata, citing the source (Context or Metadata) for each piece of information.
7. Use neutral, factual language. Avoid subjective statements or opinions.
8. If you mention any specific fact, it MUST be present in the provided Context or Metadata.
9. Do NOT generate or hallucinate any additional information, even if it seems plausible.
10. If the question asks about something not related to the Context or Metadata, state: "The provided information is not relevant to this question."

Format your answer as follows:
Direct Answer: [Your concise, direct answer here]
Supporting Details:
- [Detail 1] (Source: [Context/Metadata])
- [Detail 2] (Source: [Context/Metadata])
...

Answer:
"""


def validate_response(response: str, metadata: dict) -> str:
    validated_response = response
    corrections = []

    # Check for date consistency
    if metadata['date'] not in response:
        corrections.append(f"The correct recording date is {metadata['date']}.")

    # Check for contributor consistency
    contributors = ", ".join(metadata['contributors'])
    if not any(contrib.lower() in response.lower() for contrib in metadata['contributors']):
        corrections.append(f"The contributors to this recording are {contributors}.")

    # Check for location consistency (if available in metadata)
    if 'locations' in metadata and metadata['locations']:
        location = metadata['locations'][0]
        if location.lower() not in response.lower():
            corrections.append(f"The recording location is {location}.")

    # Check for title consistency
    if metadata['title'].lower() not in response.lower():
        corrections.append(f"The correct title of the recording is '{metadata['title']}'.")

    if corrections:
        validated_response += "\n\nCorrections:"
        for correction in corrections:
            validated_response += f"\n• {correction}"

    return validated_response


def structure_response(response: str) -> str:
    parts = response.split("\n\nCorrections:")
    main_response = parts[0]
    corrections = parts[1] if len(parts) > 1 else ""

    sentences = main_response.split('. ')
    structured_response = "RAG Response:\n\n"
    for sentence in sentences:
        structured_response += f"• {sentence.strip()}.\n"

    if corrections:
        structured_response += f"\nCorrections:{corrections}"

    return structured_response


def integrate_metadata(response: str, metadata: dict) -> str:
    relevant_fields = ['title', 'date', 'contributors', 'subjects', 'type', 'url']
    metadata_section = "Relevant Metadata:\n"

    for field in relevant_fields:
        if field in metadata and metadata[field]:
            value = metadata[field] if isinstance(metadata[field], str) else ', '.join(metadata[field])
            metadata_section += f"• {field.capitalize()}: {value}\n"

    return f"{metadata_section}\n{response}"