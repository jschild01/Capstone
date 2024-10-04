from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


class RAGGenerator:
    def __init__(self, model_name, hf_token):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name) # model_name='google/flan-t5-base'
        #self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_response(self, query: str, context: str, max_length: int = 150) -> str:
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        input_ids = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).input_ids

        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()