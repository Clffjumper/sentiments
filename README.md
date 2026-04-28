!pip install -q transformers datasets evaluate accelerate

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np

dataset = load_dataset("imdb")
# Use smaller subset for faster training (important for demo)
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(2000))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)



tokenized_datasets = dataset.map(preprocess_function, batched=True)



data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


accuracy = evaluate.load("accuracy")



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)




training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
)




trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
  
    data_collator=data_collator,
    compute_metrics=compute_metrics
)





trainer.train()



trainer.evaluate()




from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier(""" Sorry"""))
