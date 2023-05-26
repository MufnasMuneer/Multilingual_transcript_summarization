import pandas as pd
import numpy as np
import csv

# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load pre-trained model and tokenizer
device = torch.device('cuda')
cpu = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('./models/model3989')
model = AutoModelForSeq2SeqLM.from_pretrained('./models/model3989')
model = model.to(device)
model.eval()

df_test = pd.read_csv('test_cleaned_en-hin_9999.csv')
# Convert dataframe to list of dictionaries
data = df_test.to_dict('records')

# Define dataset and dataloader classes
class DocumentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    inputs = tokenizer([x['utt'] for x in batch], return_tensors='pt', padding=True, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, do_sample=False)
    summary_ids = summary_ids.to(cpu)
    summaries = [tokenizer.decode(summary_ids[i], skip_special_tokens=True) for i in range(len(summary_ids))]
    for i, summary in enumerate(summaries):
        batch[i]['summary'] = summary
    return batch

# Example usage
dataset = DocumentDataset(data)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

with open('predicted_sum-50671120937.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['text', 'summary'])
    for batch in tqdm(dataloader):
        for document in batch:
            writer.writerow([document['utt'], document['summary']])