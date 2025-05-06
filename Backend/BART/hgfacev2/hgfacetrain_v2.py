from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import transformers


# 1. Carrega o dataset CSV
dataset = load_dataset('csv', data_files={'train': r'C:\Users\RLima\OneDrive\Documentos\VS CODE\APS - NLP\APS---NLP\Backend\treinonoticiasv2.csv'}, delimiter=',')

# 2. Verifica e converte o label para inteiro, se necessário
def label_to_int(example):
    label_map = {'negativo': 0, 'neutro': 1, 'positivo': 2}
    example['label'] = label_map[example['label']]
    return example

dataset = dataset.map(label_to_int)

# 3. Carrega o tokenizer do modelo previamente treinado
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\RLima\OneDrive\Documentos\VS CODE\APS - NLP\APS---NLP\Backend\modelo_finetunado_bert_pt")

# 4. Tokeniza os textos
def preprocess_function(examples):
    return tokenizer(examples['texto'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset['train'].map(preprocess_function, batched=True)

# 5. Carrega o modelo previamente treinado
model = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\RLima\OneDrive\Documentos\VS CODE\APS - NLP\APS---NLP\Backend\modelo_finetunado_bert_pt")

# 6. Defina os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results_bert_pt_v2",
    eval_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_strategy="no",
    logging_dir='./logs_bert_pt_v2',
    logging_steps=100,
)

# 7. Função de métricas
def compute_metrics(eval_pred):
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
trainer.save_model("./modelo_finetunado_bert_pt_v2")
tokenizer.save_pretrained("./modelo_finetunado_bert_pt_v2")