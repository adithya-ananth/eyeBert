from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Hemanth-thunder/english-tamil-mt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "I, Amarendra Bahubali, swear to protect the wealth, respect and lives of the people of Mahishmathi"

inputs = tokenizer.encode(text, return_tensors="pt")

outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)

translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Translated text:", translated_text)
