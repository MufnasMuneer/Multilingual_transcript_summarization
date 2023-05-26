import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import evaluate

df=pd.read_csv('train_cleaned_en-hin_40000.csv')
device = torch.device('cuda')
cpu_device = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-cc25")
model = model.to(device)

# Define your custom dataset class
class CustomTranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, index):
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]

        source_encoded = self.tokenizer.encode(source_text, padding=True, truncation=True, max_length=1024)
        target_encoded = self.tokenizer.encode(target_text, padding=True, truncation=True, max_length=1024)
        return {
            'input_ids': source_encoded,
            'attention_mask': [1] * len(source_encoded),
            'labels': target_encoded,
        }

train_size = 32000
train_dataset = CustomTranslationDataset(df['utt'][:train_size].tolist(), df['summary'][:train_size].tolist(), tokenizer)
val_dataset = CustomTranslationDataset(df['utt'][train_size:].tolist(), df['summary'][train_size:].tolist(), tokenizer)
del df


rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

training_args = TrainingArguments(
    output_dir='./working/',
    overwrite_output_dir=False,
    num_train_epochs=1,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size for training
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,  # Batch size for evaluation
    eval_accumulation_steps=10,
    save_steps=1000,  # Number of steps to save the model during training
    save_total_limit=2,  # Maximum number of saved checkpoints
    evaluation_strategy='epoch',  # Evaluation strategy (epoch, steps, no)
    logging_dir='./working/logs',  # Directory to save training logs
    # do_train=False, do_eval=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the Trainer for training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics = compute_metrics
)

# Start training
trainer.train()

# trainer.evaluate()

model.save_pretrained("saved_model_32000_10000")