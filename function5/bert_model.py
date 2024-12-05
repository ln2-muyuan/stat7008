import pandas as pd
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 检查CUDA可用性
if torch.cuda.is_available():
    device = "cuda"
    print("GPU is available，Training on GPU")
else:
    device = "cpu"
    print("GPU is not available. Training on CPU")

# 1. 读取数据
df = pd.read_csv('ESG_Score_Conclusion.csv', encoding='ISO-8859-1')
texts = df['ESG Report Conclusion'].tolist()
labels = df['ESG Rating'].tolist()

# 2. 编码类别
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 3. 分割数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42)

# 4. 使用BERT tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 5. 创建Dataset类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids  # 获取真实标签
    preds = pred.predictions.argmax(-1)  # 获取模型预测的类别
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')  # 计算精确率、召回率和F1-score
    acc = accuracy_score(labels, preds)  # 计算准确率
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 6. 准备数据集
train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)

# 7. 加载BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
for param in model.bert.encoder.layer[-1:].parameters():
    param.requires_grad = True
# 8. 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=80,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy='no',
)

# 9. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 10. 训练模型
trainer.train()

# 11. 进行预测
predictions = trainer.predict(val_dataset)
predicted_labels = predictions.predictions.argmax(-1)

# 12. 输出预测结果
print("预测标签:", label_encoder.inverse_transform(predicted_labels))
print("真实标签:", label_encoder.inverse_transform(val_labels))

"""
Result:

All of the accuracy, F1 score, precision, recall do not perform well in this model. 

Final Result:
{'eval_loss': 2.7519428730010986, 'eval_accuracy': 0.4074074074074074, 'eval_f1': 0.407631874298541, 
'eval_precision': 0.4388888888888889, 'eval_recall': 0.4074074074074074, 'eval_runtime': 0.1112, 
'eval_samples_per_second': 242.863, 'eval_steps_per_second': 17.99, 'epoch': 80.0}
{'train_runtime': 204.9397, 'train_samples_per_second': 41.768, 'train_steps_per_second': 2.733, 
'train_loss': 0.5467348856851458, 'epoch': 80.0} 

But if we accept some small error, for example we consider that predict AAA to AA is acceptable, we will get a higher performance
to eval_accuracy: 0.6296296296296297

"""