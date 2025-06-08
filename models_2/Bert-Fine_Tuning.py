import json
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

train_file = '/content/sample_data/train.json'
eval_file = '/content/sample_data/test.json'

with open(train_file, 'r') as f:
    train_data = json.load(f)

with open(eval_file, 'r') as f:
    eval_data = json.load(f)

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return sum(len(item['qas']) for item in self.data)

    def __getitem__(self, idx):
        for item in self.data:
            if idx < len(item['qas']):
                qa_pair = item['qas'][idx]
                break
            idx -= len(item['qas'])

        context = item['context']
        question = qa_pair['question']
        answers = qa_pair['answers']

        inputs = self.tokenizer(
            question, context,
            max_length=self.max_length, truncation=True, padding="max_length", return_offsets_mapping=True, return_tensors="pt"
        )

        offset_mapping = inputs.pop("offset_mapping")[0]

        start_char = answers[0]['answer_start']
        end_char = start_char + len(answers[0]['text'])

        token_start_index = None
        token_end_index = None

        for i, (start, end) in enumerate(offset_mapping):
            if start <= start_char < end:
                token_start_index = i
            if start < end_char <= end:
                token_end_index = i
                break

        if token_start_index is None:
            token_start_index = 0 
        if token_end_index is None:
            token_end_index = inputs['input_ids'].shape[0] - 1

        inputs.update({
            'start_positions': torch.tensor(token_start_index, dtype=torch.long),
            'end_positions': torch.tensor(token_end_index, dtype=torch.long)
        })

        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)

        return inputs

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
train_dataset = QADataset(train_data, tokenizer)
eval_dataset = QADataset(eval_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

model = BertForQuestionAnswering.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8, 
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs', 
    report_to="none", 
    do_eval=True, 
    eval_steps=500  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset  
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

model.save_pretrained("./fine-tuned-bert")
tokenizer.save_pretrained("./fine-tuned-bert")
