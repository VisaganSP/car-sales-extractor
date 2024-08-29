import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

# Define constants
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128

# Custom dataset class
class CarSalesDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.label_map = self.create_label_map()

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            return json.load(f)

    def create_label_map(self):
        # Define labels - This should be aligned with your data's labels
        labels = ['O', 'B-CAR_TYPE', 'I-CAR_TYPE', 'B-FUEL_TYPE', 'I-FUEL_TYPE',
                  'B-COLOR', 'I-COLOR', 'B-MAKE_YEAR', 'I-MAKE_YEAR',
                  'B-TRANSMISSION_TYPE', 'I-TRANSMISSION_TYPE',
                  'B-CAR_MAKE', 'I-CAR_MAKE', 'B-CAR_MODEL', 'I-CAR_MODEL']
        return {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['prompt']
        labels = self.encode_labels(sample['completion'], text)

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        # Align labels with tokens
        labels_encoded = self.align_labels_with_tokens(encoding, labels)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels_encoded)
        }

    def encode_labels(self, completion, text):
        # Custom function to map completion keys to text positions and generate labels
        # Example of how to convert completion dict to a label sequence
        labels = ['O'] * len(text.split())  # Start with 'O' for 'Outside'
        if completion.get('Car Type'):
            labels = self.set_labels(labels, text, completion['Car Type'], 'CAR_TYPE')
        if completion.get('Fuel Type'):
            labels = self.set_labels(labels, text, completion['Fuel Type'], 'FUEL_TYPE')
        if completion.get('Color'):
            labels = self.set_labels(labels, text, completion['Color'], 'COLOR')
        if completion.get('Transmission Type'):
            labels = self.set_labels(labels, text, completion['Transmission Type'], 'TRANSMISSION_TYPE')
        if completion.get('Car Make'):
            labels = self.set_labels(labels, text, completion['Car Make'], 'CAR_MAKE')
        if completion.get('Car Model'):
            labels = self.set_labels(labels, text, completion['Car Model'], 'CAR_MODEL')
        if completion.get('Make Year'):
            labels = self.set_labels(labels, text, completion['Make Year'], 'MAKE_YEAR')

        return labels

    def set_labels(self, labels, text, value, entity_type):
        words = text.split()
        entity_words = value.split()
        for i, word in enumerate(words):
            if word in entity_words:
                labels[i] = f'B-{entity_type}' if i == 0 else f'I-{entity_type}'
        return labels

    def align_labels_with_tokens(self, encoding, labels):
        # Align labels to tokens (for BERT sub-word tokens)
        label_ids = []
        word_ids = encoding.word_ids()
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignored in loss calculation
            else:
                label_ids.append(self.label_map[labels[word_id]])
        return label_ids

# Load tokenizer and create dataset
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = CarSalesDataset('data/training_data/training_data.json', tokenizer)

# Data collator for token classification
data_collator = DataCollatorForTokenClassification(tokenizer)

# Load pre-trained BERT model
model = BertForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(train_dataset.label_map)
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_steps=500,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('models/bert_model')
tokenizer.save_pretrained('models/bert_model')

# Load LLaMA 2 (7B) Model and Tokenizer
llama_model_name = 'llama2-7b'
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)

def process_text_with_bert(text):
    """Process input text using BERT and return embeddings."""
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()  # Average pooling

def generate_text_with_llama2(embedding):
    """Generate text using LLaMA 2 model based on BERT embeddings."""
    # Convert BERT embeddings to text prompt
    # For this example, we assume embeddings are transformed into a prompt
    prompt = f"Use the following information to generate text: {embedding.tolist()}"
    inputs = llama_tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = llama_model.generate(**inputs, max_length=100)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    text = "Your input text here."
    
    # Step 1: Process text with BERT
    embedding = process_text_with_bert(text)
    
    # Step 2: Generate text with LLaMA 2 based on BERT embeddings
    generated_text = generate_text_with_llama2(embedding)
    
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
