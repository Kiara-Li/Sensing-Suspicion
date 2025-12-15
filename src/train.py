import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
# 确保 preprocess.py 在同一目录下，或者在 PYTHONPATH 中
from preprocess import clean_text
import os

# Mac M1/M2/M3 优化配置
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_metrics(p):
    """
    计算评估指标：准确率、召回率、F1分数
    让我们可以直观看到模型是不是在变聪明
    """
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='binary')
    precision = precision_score(y_true=labels, y_pred=pred, average='binary')
    f1 = f1_score(y_true=labels, y_pred=pred, average='binary')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_model():
    # 1. 加载数据
    data_path = "data/raw/reddit_posts.csv"
    if not os.path.exists(data_path):
        print("Data not found. Please run scrape_reddit.py first.")
        return

    print("Loading and processing data...")
    df = pd.read_csv(data_path)
    
    # 强制转换为字符串并清洗
    df['text'] = df['text'].astype(str).apply(clean_text)
    
    # 再次检查：去掉空文本
    df = df[df['text'].str.len() > 10]

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    print(f"Training on {len(train_texts)} samples, Validating on {len(val_texts)} samples.")

    # 2. Tokenization
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def tokenize_function(texts):
        # max_length 设为 256 或 512。如果不缺显存/内存，设为 512 效果最好
        return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    # 创建 Dataset 对象
    class RedditDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = RedditDataset(train_encodings, train_labels)
    val_dataset = RedditDataset(val_encodings, val_labels)

    # 3. 加载模型
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4. 训练参数 (已针对 Mac 和新数据优化)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,              # 增加轮数，让模型学透
        per_device_train_batch_size=8,   # 如果内存报错，改小成 4
        per_device_eval_batch_size=8,
        learning_rate=2e-5,              # 这是一个比较稳健的学习率
        warmup_steps=100,                # 热身步数
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        
        # 评估策略：每个 epoch 结束测一次
        eval_strategy="epoch",           
        save_strategy="epoch",
        
        # 关键：加载最好的模型，而不是最后一步的模型
        load_best_model_at_end=True,     
        metric_for_best_model="accuracy" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics  # 传入上面的计算函数
    )

    # 5. 开始训练
    print("Starting training...")
    trainer.train()

    # 6. 保存模型
    save_path = "models/creepy_roberta"
    print(f"Saving best model to {save_path}...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done!")

    # 7. 打印最终验证集结果
    print("\n--- Final Evaluation on Validation Set ---")
    eval_result = trainer.evaluate()
    print(eval_result)

if __name__ == "__main__":
    train_model()