import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('./mediasum453k.csv')
df = df.rename(columns={'utt': 'input_text', 'summary': 'target_text'})

# Drop rows with missing values
df.dropna(inplace=True)

# Convert text to lowercase
# df['input_text'] = df['input_text'].str.lower()

for i in range(len(df)):
  df['utt'][i]=df['utt'][i].replace('/', "").replace('\\', "").replace('"', '').replace('\'','')

# Remove links
df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'http\S+', '', x))

# Remove special characters and digits
df['input_text'] = df['input_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
df['input_text'] = df['input_text'].str.replace(r'\d+', '', regex=True)

df.to_csv('./cleaned_eng.csv', index=False, header=True, encoding='utf-8')

