import torch
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("squad")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
def preprocess_data(examples):
    inputs = tokenizer(
        examples['question'], examples['context'], 
        max_length=384, truncation=True, padding="max_length", return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i in range(len(examples['answers']['text'])):
        start_char = examples['answers']['answer_start'][i]
        end_char = start_char + len(examples['answers']['text'][i])

        token_start_index = inputs.char_to_token(i, start_char)
        token_end_index = inputs.char_to_token(i, end_char - 1)

        if token_start_index is None or token_end_index is None:
            token_start_index = tokenizer.model_max_length
            token_end_index = tokenizer.model_max_length

        start_positions.append(token_start_index)
        end_positions.append(token_end_index)

    inputs.update({
        'start_positions': start_positions,
        'end_positions': end_positions
    })

    return inputs

train_dataset = dataset['train'].map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
eval_dataset = dataset['validation'].map(preprocess_data, batched=True, remove_columns=dataset['validation'].column_names)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

trainer.evaluate()
model.save_pretrained("./fine-tuned-bert")
tokenizer.save_pretrained("./fine-tuned-bert")
