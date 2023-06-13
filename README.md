Multilingual Transcript Summarization
Introduction
With the exponential growth of internet content globally, the need for efficient summarization techniques has become paramount. Transcript summarization plays a crucial role in distilling essential information from large volumes of data, especially when the content contains redundant or repetitive information. This process saves time and effort by providing a concise and understandable format, making it invaluable for various domains such as education, legal processing, and market research.

Objective
This project aims to propose an abstractive multilingual transcript summarization model by leveraging state-of-the-art transformers, specifically mBART and mT5, to target the English and Hindi languages. While recent works in summarization predominantly focus on the English language due to the abundance of resources available, our project addresses the significant challenge posed by low-resource languages like Hindi.

Methodology
To overcome the scarcity of summarization datasets for Hindi, we have taken a two-fold approach. Firstly, we create a Hindi dataset by translating the existing MediaSUM dataset using cutting-edge Machine Translation techniques. This step ensures that we have a representative dataset for the Hindi language. Secondly, we combine this newly generated MediaSUM dataset for Hindi with the existing English dataset to form a bilingual corpus.

The transformers, including mBART and mT5, are then fine-tuned on this bilingual corpus, allowing them to learn and capture the nuances of both English and Hindi languages. By benchmarking and evaluating the performance of these models, we aim to identify the best-performing approach for multilingual transcript summarization.

Benefits and Applications
The proposed multilingual transcript summarization model offers several benefits and applications:

Time and Effort Savings: By distilling essential information from large amounts of data, the model saves time and effort for individuals or organizations processing vast quantities of transcripts.

Education: The summarized transcripts can be used in educational settings to provide concise and easily understandable material for students, facilitating effective learning.

Legal Processing: Legal professionals can benefit from transcript summarization when reviewing case-related information, extracting key details, and identifying relevant evidence more efficiently.

Market Research: Companies engaged in market research can leverage transcript summarization to extract valuable insights from interviews, surveys, and customer feedback, enabling informed decision-making.

Conclusion
The proposed abstractive multilingual transcript summarization model, fine-tuned on a bilingual corpus of English and Hindi, addresses the need for efficient and effective summarization in both resource-rich and low-resource languages. By leveraging state-of-the-art transformers like mBART and mT5, we aim to provide a solution that aids in extracting essential information from transcripts across diverse domains. The benefits and applications of this approach extend to education, legal processing, market research, and beyond, offering time and effort savings while maintaining the quality and relevance of the summarized content.

-- could not add model to git due to 6.5GB size
Model can be viewed here : https://drive.google.com/file/d/1a7swYwd0CZSKau5NP1yTXwLJZU6P1Cyy/view?ts=64498049
