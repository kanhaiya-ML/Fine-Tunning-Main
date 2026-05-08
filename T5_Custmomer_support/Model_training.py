from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
import torch
import json

FILE_PATH = r"C:\Users\Kanhaiya\OneDrive\Desktop\projects\customer_complian\data.json"
with open(FILE_PATH,"r") as f:
    data = json.load(f)
#spliting data first
train_data,val_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

#convert to huggingface dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# print(train_dataset)
# print(val_dataset)

tokenizer = AutoTokenizer.from_pretrained(
    "google-t5/t5-small"
    )

def tokenize(example):
    model_input = tokenizer(
        example['input'],
        max_length=64,
        truncation=True,
        padding='max_length'
    )

    label = tokenizer(
        example['output'],
        max_length=128,
        truncation=True,
        padding='max_length'
    )

    labels = label['input_ids'].copy()
    labels = [
        -100 if token == tokenizer.pad_token_id 
        else token 
        for token in labels
    ]
    model_input['labels'] = labels
    return model_input

#apply to dataset
train_tokenized = train_dataset.map(tokenize)
val_tokenized = val_dataset.map(tokenize)

# print(train_tokenized)


model = AutoModelForSeq2SeqLM.from_pretrained(
    "google-t5/t5-small"
)
training_args = Seq2SeqTrainingArguments(
    output_dir='/customer_complaint',
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-4,
    eval_strategy='epoch',
    save_strategy='epoch',
    predict_with_generate=True,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
trainer.train()

def predict(text):
    # Add prefix
    input_text = "customer support: " + text
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        max_length=64,
        truncation=True
    )

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode output
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    
    print(f"Customer: {text}")
    print(f"Bot: {response}\n")

# Test
predict("my payment failed but money was deducted")
predict("i want to exchange my product")
predict("where is my order")
predict("what is the weather today")  # off-topic test