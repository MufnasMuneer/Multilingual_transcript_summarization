
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np


# clone the repo for running evaluation
!git clone https://github.com/AI4Bharat/indicTrans.git
%cd indicTrans
# clone requirements repositories
!git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
!git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
!git clone https://github.com/rsennrich/subword-nmt.git
%cd ..

# Install the necessary libraries
!pip install sacremoses pandas mock sacrebleu tensorboardX pyarrow indic-nlp-library
! pip install mosestokenizer subword-nmt
# Install fairseq from source
!git clone https://github.com/pytorch/fairseq.git
%cd fairseq
# !git checkout da9eaba12d82b9bfc1442f0e2c6fc1b895f4d35d
!pip install ./
! pip install xformers
%cd ..



# add fairseq folder to python path
import os
os.environ['PYTHONPATH'] += ":/content/fairseq/"
# sanity check to see if fairseq is installed
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils


# download the indictrans model


# downloading the indic-en model
!wget https://storage.googleapis.com/samanantar-public/V0.3/models/indic-en.zip
!unzip indic-en.zip

# downloading the en-indic model
!wget https://storage.googleapis.com/samanantar-public/V0.3/models/en-indic.zip
!unzip en-indic.zip

# # downloading the indic-indic model
!wget https://storage.googleapis.com/samanantar-public/V0.3/models/m2m.zip
!unzip m2m.zip

%cd indicTrans



# Load the indicTrans model for English to Hindi translation
from indicTrans.inference.engine import Model

en2indic_model = Model(expdir='/content/drive/MyDrive/ANLP/IndicTransmodel/en_indic/en-indic')


df=pd.read_csv('/content/drive/MyDrive/ANLP/cleaned_eng.csv')


summary=[]
utt=[]

#  Translate each row in the DataFrame from English to Hindi
# Going in steps of 25000 to avoid 
x_high = 425000
x_low = 400000
for i in range(x_low, x_high):
  print("Iteration :", i)
  hin_utt=en2indic_model.translate_paragraph(df['utt'][i], 'en', 'hi')
  utt.append(hin_utt)
  hin_summary=en2indic_model.translate_paragraph(df['summary'][i], 'en', 'hi')
  summary.append(hin_summary)

len(summary)


hindi_df={'utt': utt[:822],
          'summary': summary[:822]}

hindi_df=pd.DataFrame(hindi_df)

hindi_df.to_csv('/content/drive/MyDrive/ANLP/25k.csv', index=False, header=True, encoding='utf-8')

