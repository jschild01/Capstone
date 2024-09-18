from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class QAGenerator:
    def __init__(self, model_name='google/flan-t5-small'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_response(self, query, most_relevant_passage, max_new_tokens=100):
        input_text = f"Question: {query}\nContext: {most_relevant_passage}\nAnswer:"
        input_ids = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).input_ids
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            temperature=1.0,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
