from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import boto3
from botocore.exceptions import ClientError
import json
import os
from dotenv import load_dotenv
from configparser import ConfigParser, ExtendedInterpolation
import sys


class RAGGenerator:
    def __init__(self, model_name):
        if model_name == 'llama':
            model_name = 'meta-llama/Llama-3.2-3B-Instruct'
            hf_token = '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'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        elif model_name == 't5':
            model_name = 'google/flan-t5-xl'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif model_name == 'claude':
            #model_name = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
            self.config = self.load_configuration()
            #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            #self.model = AutoModelForCausalLM.from_pretrained(model_name)
            #self.model = model_name
            self.bedrock_client = self.create_bedrock_client(self.config)
        else:
            raise ValueError("Invalid model name provided. Input either 'llama', 'claude', or 't5' as model name.")
        
    def load_configuration(self):
        # Set the current working directory to the project root
        components_dir = os.path.dirname(__file__)
        src_dir = os.path.abspath(os.path.join(components_dir, os.pardir))
        root_dir = os.path.abspath(os.path.join(src_dir, os.pardir))
        config_dir = os.path.join(root_dir, 'config')

        load_dotenv(dotenv_path=os.path.join(config_dir, '.env'))
        config_file = os.environ['CONFIG_FILE']
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(f"{config_dir}/{config_file}")
        return config

    def create_bedrock_client(self, config):
        session = boto3.Session(
            aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
            aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
            aws_session_token=config['BedRock_LLM_API']['aws_session_token']
        )
        return session.client("bedrock-runtime", region_name="us-east-1")

    def generate_response(self, prompt: str, max_length: int = 300) -> str:
        if hasattr(self, 'tokenizer') and hasattr(self, 'model'):
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2000)

            # Set attention mask
            attention_mask = inputs['attention_mask']

            # Set pad token ID
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()

        elif hasattr(self, 'bedrock_client'):
            # Update the dictionary to match the expected API schema
            request_payload = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_length,  # Corrected key from max_tokens_to_sample to max_tokens
                "temperature": 0.7
            }).encode('utf-8')

            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=request_payload,
                contentType='application/json'
            )

            # Decode the response assuming it returns JSON with 'generated_text' key
            response_body = response['body'].read().decode()
            return response_body
            #response_body = json.loads(response['body'].read().decode())
            #return response_body['messages'][0]['content']
            #return json.loads(response['body'])['generated_text']
        
        '''
        elif hasattr(self, 'bedrock_client'):
            # Serialize the dictionary to JSON and encode it to bytes
            serialized_input = json.dumps({'input_text': prompt, 'max_tokens': max_length, 'temperature': 0.7}).encode('utf-8')
            response = self.bedrock_client.invoke_model(
                modelId=self.model,
                body=serialized_input,  # Pass the byte-encoded JSON string
                contentType='application/json'
            )
            # Assuming the response is JSON, decode and parse it
            return json.loads(response['body'])['generated_text']
        
        elif hasattr(self, 'bedrock_client'):
            response = self.bedrock_client.invoke_model(
                modelId=self.model,
                body={'input_text': prompt, 'max_tokens': max_length, 'temperature': 0.7},
                contentType='application/json'
            )
            return response['body']['generated_text']
        '''

