import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load the datasets
df = pd.read_csv("https://raw.githubusercontent.com/causalNLP/logical-fallacy/refs/heads/main/data/climate_all.csv")
df1 = pd.read_csv("https://raw.githubusercontent.com/tmakesense/logical-fallacy/refs/heads/main/original-logical-fallacy-by-causalNLP/data/edu_all.csv")

# Extract and rename columns to be consistent
df_cleaned = df[['source_article', 'logical_fallacies']].rename(columns={
    'source_article': 'sentences',
    'logical_fallacies': 'Logical_fallacies'
})

df1_cleaned = df1[['source_article', 'updated_label']].rename(columns={
    'source_article': 'sentences',
    'updated_label': 'Logical_fallacies'
})

# Merge both datasets into one
merged_df = pd.concat([df_cleaned, df1_cleaned], ignore_index=True)

# Filter and keep only the 5 specified fallacies
allowed_classes = [
    'faulty generalization',
    'intentional',
    'ad hominem',
    'appeal to emotion',
    'ad populum'
]

merged_df = merged_df[merged_df['Logical_fallacies'].isin(allowed_classes)]
print(merged_df)
# ======= BALANCE DATA BY UNDERSAMPLING =======
min_samples = 295  # target samples per class

balanced_df = pd.DataFrame()
for fallacy in allowed_classes:
    class_samples = merged_df[merged_df['Logical_fallacies'] == fallacy]
    if len(class_samples) > min_samples:
        class_samples = class_samples.sample(min_samples, random_state=42)  # random undersample
    balanced_df = pd.concat([balanced_df, class_samples], ignore_index=True)

df = balanced_df[['sentences', 'Logical_fallacies']]
df.dropna(inplace=True)
# ======= END BALANCE STEP =======

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Logical_fallacies'])

# Print class distribution to check for imbalance
print("Class distribution after balancing:")
print(df['Logical_fallacies'].value_counts())

# Compute class weights for imbalanced classes, capping weights for stability
class_counts = df['label'].value_counts().sort_index()
total_samples = len(df)
class_weights = torch.tensor([min(total_samples / (len(class_counts) * count), 10.0) for count in class_counts], dtype=torch.float)  # Cap weights at 10

# Split data
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Custom Dataset
class FallacyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.texts = dataframe['sentences'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Initialize tokenizer and model
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
model = DebertaForSequenceClassification.from_pretrained(
    'microsoft/deberta-base',
    num_labels=len(label_encoder.classes_)
)

# Create datasets
train_dataset = FallacyDataset(train_df, tokenizer)
eval_dataset = FallacyDataset(eval_df, tokenizer)

# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

# Custom Trainer with class-weighted loss, handling num_items_in_batch
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        device = logits.device
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to='none'
)

# Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fallacy_model')
tokenizer.save_pretrained('./fallacy_model')

# Example prediction function
def predict_fallacy(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer(text, return_tensors='pt', max_length=256, padding='max_length', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(-1).item()
    return label_encoder.inverse_transform([prediction])[0]

# Test predictions
test_texts = [
"The scientist made that claim just to mislead the public and push a secret agenda.",
"He spread false rumors on purpose to damage his rival’s reputation.",
"She exaggerated the facts intentionally to get more support for her cause.",
"The politician lied during the debate to manipulate voters.",
"They designed the advertisement to trick customers into buying useless products.",
]
for text in test_texts:
    print(f"Text: {text}")
    print(f"Predicted fallacy: {predict_fallacy(text)}\n")
