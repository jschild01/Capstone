from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


class RAGGenerator:
    def __init__(self, model_name):
        
        if model_name=='llama3':
            #model_name = 'meta-llama/Llama-3.1-70B-Instruct'
            model_name = 'meta-llama/Llama-3.2-3B-Instruct'
            #model_name = 'meta-llama/Llama-3.2-1B-Instruct' 
            hf_token = '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' # huggingface key req'd for llama model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        elif model_name=='t5':
            model_name='google/flan-t5-large'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError("Invalid model name provided. Input either 'llama' or 't5' as model name.")            

    def generate_response(self, query: str, context: str, max_length: int = 150) -> str:
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        input_ids = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).input_ids

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=100,
            #max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
