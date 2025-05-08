from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import transformers


# 1. Carrega o dataset CSV
from sklearn.model_selection import train_test_split
import pandas as pd

# Carrega o CSV usando pandas
csv_path = 'Backend/BART/newdataset.csv'
df = pd.read_csv(csv_path)

# Converte o label para inteiro
label_map = {'negativa': 0, 'irrelevante': 1, 'positiva': 2}
df['label'] = df['label'].map(label_map)

# Separa em treino e validação (80/20)
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Salva temporariamente os splits
train_path = 'Backend/BART/train_tmp.csv'
val_path = 'Backend/BART/val_tmp.csv'
df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path, index=False)

# Carrega os datasets com Huggingface
from datasets import load_dataset

dataset = load_dataset('csv', data_files={'train': train_path, 'validation': val_path}, delimiter=',')vante': 1, 'positiva': 2}
df['label'] = df['label'].map(label_map)

# Separa em treino e validação (80/20)
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Salva temporariamente os splits
train_path = 'Backend/BART/train_tmp.csv'
val_path = 'Backend/BART/val_tmp.csv'
df_train.to_csv(train_path, index=False)
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. Carrega o modelo para classificação multiclasse
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

dataset = load_dataset('csv', data_files={'train': train_path, 'validation': val_path}, delimiter=',')
training_args = TrainingArguments(
    output_dir="./results_bert_pt_final",
    evaluation_strategy="epoch",
    num_train_epochs=8,  # aumente conforme necessário
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_dir='./logs_bert_pt',
    logging_steps=100,
tokenized_dataset = dataset['train'].map(preprocess_function, batched=True)

# 5. Carrega o modelo para classificação binária
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 6. Defina os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results_bert_pt_final",
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
# 8. Treinador
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_weighted': f1_score(labels, predictions, average='weighted')
    }

# 8. Treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Ideal: separar um conjunto de validação
    compute_metrics=compute_metrics,
)

# 9. Treinamento
trainer.train()

# 10. Salve o modelo ajustado
trainer.save_model("./bart_tunado")
tokenizer.save_pretrained("./bart_tunado")