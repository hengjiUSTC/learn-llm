import csv
import pandas as pd
import requests
from datasets import Dataset

def load_json_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to load data from URL")

def generate_prompt(data_point):
    tags = ';'.join(data_point['tags'])
    paragraph = '\n'.join(data_point['paragraphs'])
    return f"""<s>你是一个唐诗助手,帮助用户写一首对应要求的唐诗

INPUT:
作者:{data_point["author"]}
标签:{tags}

OUTPUT:
{data_point['title']}
{paragraph}</s>
""".strip()

def generate_text(data_point):
    full_prompt = generate_prompt(data_point)
    return {"text": full_prompt}

# URL of the JSON file
url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/%E5%94%90%E8%AF%97%E4%B8%89%E7%99%BE%E9%A6%96.json"
data = load_json_from_url(url)

# Convert data to Pandas DataFrame
df = pd.DataFrame(data=data)

# Create Dataset and apply transformations
dataset = Dataset.from_pandas(df)
dataset = dataset.shuffle().map(generate_text)

train_test_split = dataset.train_test_split(test_size=0.1)

# You now have 80% train, 10% test, and 10% validation sets
train_set = train_test_split['train']
validation_set = train_test_split['test']

output_file = 'train.csv'

# Open the file in write mode
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['text'])

    # Iterate over the dataset and write the 'text' field
    for row in train_set:
        writer.writerow([row['text']])


output_file = 'val.csv'

# Open the file in write mode
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['text'])

    # Iterate over the dataset and write the 'text' field
    for row in validation_set:
        writer.writerow([row['text']])
