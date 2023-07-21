import csv
import random
from alignscore import AlignScore
import sys 
import pandas as pd
# Set the field size limit to a larger value
csv.field_size_limit(sys.maxsize)

ckpt_path = 'https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt'

# ckpt_path = 'https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt'
scorer = AlignScore(model='roberta-base', batch_size=32, device='cpu', ckpt_path=ckpt_path, evaluation_mode='nli_sp')


# # Read the original CSV file
# input_file = "now_data.csv"
# output_file = "now_data_sampled_scored_.csv"
# num_rows_to_extract = 100
# seed_value = 42  # Set your desired seed value here

# # Set the seed value for random.sample()
# random.seed(seed_value)

# data = []
# with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         data.append(row)

# # Randomly extract 100 rows
# random_data = random.sample(data, num_rows_to_extract)

# print("data ready")
# # Calculate scores for each row and add them to a new column
# for row in random_data:
#     passage = [row["source_text"]]
#     claim = [row["statement"]]
#     row["score"] = score = scorer.score(contexts=passage, claims=claim)

# # Save the new 100 rows to a new CSV file
# with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#     fieldnames = list(random_data[0].keys()) #+ ["score"]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerows(random_data)






output_file = "train_data_with_Alignscore.csv"
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np


# Custom function to split data based on the "query" field
def custom_train_test_split(data, test_size=0.25, random_state=42):
    unique_queries = data['query'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_queries)

    train_queries = unique_queries[:int(len(unique_queries) * (1 - test_size))]
    val_queries = unique_queries[int(len(unique_queries) * (1 - test_size)):]

    train_data = data[data['query'].isin(train_queries)]
    val_data = data[data['query'].isin(val_queries)]

    return train_data, val_data

data = pd.read_csv("now_data.csv")
# Assuming 'data' is your original DataFrame
train_data, val_data = custom_train_test_split(data, test_size=0.25, random_state=42)
data = train_data
print(f"Train Actual Positive Samples: {sum(train_data['detection_label'] == True)}. Among {len(train_data)} Samples")
print(f"Val Actual Positive Samples: {sum(val_data['detection_label'] == True)}. Among {len(val_data)} Samples")

# print(val_data, type(val_data))

print("data ready")
# Calculate scores for each row and add them to a new column
i = 0
alignscores = []
for _, row in data.iterrows():
    
        
    # print(row, type(row))
    passage = [row["source_text"]]
    claim = [row["statement"]]
    score = scorer.score(contexts=passage, claims=claim)
    
    alignscores.append(score)
    print(i, len(data), score)
    i += 1
data['align_score'] = alignscores

# from IPython import embed; embed()
data.to_csv(output_file)
# Save the new 100 rows to a new CSV file
# with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#     fieldnames = list(val_data.keys()) #+ ["score"]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerows(val_data)