from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

class QAGenerator:
    def __init__(self, model_name='google/flan-t5-small'):
        self.model_name = model_name
        self.model_dict = {
            'google/flan-t5-small': (AutoTokenizer.from_pretrained, AutoModelForSeq2SeqLM.from_pretrained),
            'google/flan-t5-large': (AutoTokenizer.from_pretrained, AutoModelForSeq2SeqLM.from_pretrained),
            'gpt2': (AutoTokenizer.from_pretrained, AutoModelForCausalLM.from_pretrained)
        }
        
        # Initialize tokenizer and model based on the chosen model
        if self.model_name in self.model_dict:
            tokenizer_class, model_class = self.model_dict[self.model_name]
            self.tokenizer = tokenizer_class(self.model_name)
            self.generatorModel = model_class(self.model_name)
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

    def generate_response(self, query, most_relevant_passage, max_new_tokens=50, temperature=0.4, top_p=0.8):
        input_text = query + " " + most_relevant_passage
        input_text = input_text[:2000]
        
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output = self.generatorModel.generate(input_ids, 
                                              max_length=max_new_tokens + len(input_ids[0]), 
                                              temperature=temperature,
                                              top_p=top_p,
                                              num_return_sequences=1,
                                              do_sample=True)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return response
