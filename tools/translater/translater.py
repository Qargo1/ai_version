from transformers import MarianMTModel, MarianTokenizer

class Translater:
    def __init__(self):
        # Загрузка модели и токенизатора для перевода en -> ru
        self.model_name_en_to_ru = "Helsinki-NLP/opus-mt-en-ru"
        self.model_name_ru_to_en = "Helsinki-NLP/opus-mt-ru-en"
        self.initialize_tokinezer()
        self.initilize_model()
        
    def initialize_tokinezer(self):
        self.tokenizer_en_to_ru = MarianTokenizer.from_pretrained(self.model_name_en_to_ru)
        self.tokenizer_ru_to_en = MarianTokenizer.from_pretrained(self.model_name_ru_to_en)
        
    def initilize_model(self):
        self.model_en_to_ru = MarianMTModel.from_pretrained(self.model_name_en_to_ru)
        self.model_ru_to_en = MarianMTModel.from_pretrained(self.model_name_ru_to_en)

    def translate_ru_to_en(self, text):
        inputs = self.tokenizer_ru_to_en(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.model_ru_to_en.generate(**inputs)
        return self.tokenizer_en_to_ru.decode(translated[0], skip_special_tokens=True)
    
    def translate_en_to_ru(self, text):
        inputs = self.tokenizer_en_to_ru(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.model_en_to_ru.generate(**inputs)
        return self.tokenizer_ru_to_en.decode(translated[0], skip_special_tokens=True)
