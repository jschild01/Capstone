# test_qagenerator.py
import pytest
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import os

# Add grandparent directory since this file is in scratch_code directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from component.qa_generator import QAGenerator


@pytest.fixture
def qa_generator():
    # Fixture to create an instance of QAGenerator for use in tests
    return QAGenerator(model_name='google/flan-t5-small')

def test_generate_response_basic(qa_generator):
    # Test with a basic input
    query = "What is the capital of France?"
    passage = "France is a country in Europe. Its capital is Paris."
    response = qa_generator.generate_response(query, passage)

    # Check if the response contains the expected answer
    assert "Paris" in response, "The response did not generate the expected answer."

def test_generate_response_empty_passage(qa_generator):
    # Test when the passage is empty
    query = "What is the capital of France?"
    passage = ""
    response = qa_generator.generate_response(query, passage)

    # Check that the response is reasonable (may vary depending on the model)
    assert response != "", "The response should not be empty when passage is empty."

def test_generate_response_empty_query(qa_generator):
    # Test when the query is empty
    query = ""
    passage = "France is a country in Europe. Its capital is Paris."
    response = qa_generator.generate_response(query, passage)

    # Check that the response is reasonable (may vary depending on the model)
    assert response != "", "The response should not be empty when the query is empty."

def test_generate_response_large_passage(qa_generator):
    # Test when the passage is very long
    query = "What is the capital of France?"
    passage = " ".join(["France is a country in Europe. Its capital is Paris."] * 100)  # Simulate a long passage
    response = qa_generator.generate_response(query, passage)

    # Check if the response contains the expected answer and is not cut off
    assert "Paris" in response, "The response did not generate the expected answer for a large passage."

def test_generate_response_non_factual_query(qa_generator):
    # Test with a non-factual query
    query = "What is the best color?"
    passage = "Colors are subjective and there is no best color."
    response = qa_generator.generate_response(query, passage)

    # This checks if the model can handle non-factual queries gracefully
    assert response != "", "The response should not be empty for non-factual queries."

# Run the tests if this script is executed directly
if __name__ == "__main__":
    pytest.main()

