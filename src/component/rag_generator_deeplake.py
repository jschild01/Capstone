from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


class RAGGenerator:
    def __init__(self, model_name):
        if model_name == 'llama3':
            model_name = 'meta-llama/Llama-3.2-3B-Instruct'
            hf_token = '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'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        elif model_name == 't5':
            model_name = 'google/flan-t5-xl'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError("Invalid model name provided. Input either 'llama' or 't5' as model name.")

    def generate_response(self, prompt: str, max_length: int = 300) -> str:
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