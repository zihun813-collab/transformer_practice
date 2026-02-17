
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import os

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

if not os.path.exists("ratings_train.txt"):
    os.system("curl -O https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt")
if not os.path.exists("ratings_test.txt"):
    os.system("curl -O https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt")

df_train = pd.read_csv("ratings_train.txt", sep='\t').dropna().sample(15000, random_state=42)
df_test = pd.read_csv("ratings_test.txt", sep='\t').dropna().sample(3000, random_state=42)

train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

def preprocess_function(examples):
    return tokenizer(examples['document'], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("학습 시작")
trainer.train()

save_path = "./final_model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("저장 완료")
                                                                                                                                                                        


