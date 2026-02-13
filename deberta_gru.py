# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import _LRScheduler
# from transformers import (
#     DebertaV2Tokenizer,
#     DebertaV2Model,
#     DebertaV2Config,
#     Trainer,
#     TrainingArguments
# )
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import csv

# # 自定义MoE层
# class MoELayer(nn.Module):
#     def __init__(self, config, num_experts=4, k=2):
#         super().__init__()
#         self.num_experts = num_experts
#         self.k = k
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(config.hidden_size, config.intermediate_size),
#                 nn.ReLU(),
#                 nn.Dropout(config.hidden_dropout_prob),
#                 nn.Linear(config.intermediate_size, config.hidden_size)
#             ) for _ in range(num_experts)
#         ])
#         self.gate = nn.Sequential(
#             nn.Linear(config.hidden_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_experts)
#         )

#     def forward(self, x):
#         gate_scores = self.gate(x)
#         topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=1)
#         topk_weights = nn.functional.softmax(topk_scores, dim=1)
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         batch_indices = torch.arange(x.size(0)).unsqueeze(1).to(x.device)
#         selected_experts = expert_outputs[batch_indices, topk_indices]
#         output = (selected_experts * topk_weights.unsqueeze(-1)).sum(dim=1)
#         return output

# # 修改后的DeBERTa模型，使用GRU回归器
# class DeBERTa_MoE_Model(nn.Module):
#     def __init__(self, config, output_dim=20):
#         super().__init__()
#         self.config = config
#         self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
#         for layer in self.deberta.encoder.layer:
#             layer.ffn = MoELayer(config).to(self.deberta.device)
#         self.number_embedding = nn.Embedding(2, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.gru = nn.GRU(
#             input_size=config.hidden_size * 2,
#             hidden_size=config.hidden_size,
#             num_layers=1,
#             batch_first=True
#         )
#         self.output_layer = nn.Linear(config.hidden_size, output_dim)

#     def forward(self, input_ids, attention_mask, number_mask, labels=None):
#         outputs = self.deberta(input_ids, attention_mask=attention_mask)
#         cls_features = outputs.last_hidden_state[:, 0, :]
#         num_embed = self.number_embedding(number_mask)
#         num_features = self.dropout(num_embed.max(dim=1)[0])
#         combined = torch.cat([cls_features, num_features], dim=1).unsqueeze(1)
#         gru_out, _ = self.gru(combined)
#         output = self.output_layer(gru_out.squeeze(1))
#         if labels is not None:
#             loss = nn.functional.mse_loss(output, labels)
#             return {"loss": loss, "logits": output}
#         return {"logits": output}

# # 数据集类
# class TextDataset(torch.utils.data.Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=512):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#         tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
#         number_mask = torch.zeros(self.max_length, dtype=torch.long)
#         for i, token in enumerate(tokens):
#             if any(char.isdigit() for char in token):
#                 number_mask[i] = 1
#         return {
#             'input_ids': encoding['input_ids'].squeeze(0),
#             'attention_mask': encoding['attention_mask'].squeeze(0),
#             'number_mask': number_mask,
#             'labels': torch.tensor(label, dtype=torch.float32)
#         }

# # 自定义调度器
# class CustomCosineAnnealingWarmup(_LRScheduler):
#     def __init__(self, optimizer, num_warmup_steps, num_training_steps, eta_min=0, last_epoch=-1):
#         self.num_warmup_steps = num_warmup_steps
#         self.num_training_steps = num_training_steps
#         self.eta_min = eta_min
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         current_step = self.last_epoch + 1
#         if current_step < self.num_warmup_steps:
#             return [
#                 base_lr * current_step / self.num_warmup_steps
#                 for base_lr in self.base_lrs
#             ]
#         if current_step > self.num_training_steps:
#             return [self.eta_min for _ in self.base_lrs]
#         progress = (current_step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
#         cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
#         return [
#             self.eta_min + (base_lr - self.eta_min) * cosine_decay
#             for base_lr in self.base_lrs
#         ]

# # 数据预处理
# def preprocess_data():
#     with open('text_qwen.txt', 'r') as f:
#         text = f.readlines()
#     with open('data-augmented2.txt', 'r') as f:
#         labels = [list(map(float, line.strip().split(',')))[12:] for line in f if line.strip()]
#     text = text[:116000]
#     labels = labels[:116000]
#     scaler = StandardScaler()
#     labels = scaler.fit_transform(np.array(labels)).tolist()
#     return train_test_split(text, labels, test_size=0.2, random_state=42), scaler

# # 主程序
# def main():
#     config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-base")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     (train_texts, eval_texts, train_labels, eval_labels), scaler = preprocess_data()
#     tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
#     train_dataset = TextDataset(train_texts, train_labels, tokenizer)
#     eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer)
#     model = DeBERTa_MoE_Model(config, output_dim=20).to(device)

#     num_epochs = 15
#     dataset_size = len(train_dataset)
#     batch_size = 32
#     total_steps = (num_epochs * dataset_size) // batch_size
#     optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, eps=1e-6)
#     scheduler = CustomCosineAnnealingWarmup(optimizer, 3000, total_steps, eta_min=1e-7)

#     def compute_metrics(eval_pred):
#         predictions, labels = eval_pred
#         predictions = scaler.inverse_transform(predictions)
#         labels = scaler.inverse_transform(labels)
#         rmse = np.sqrt(mean_squared_error(labels, predictions))
#         return {"rmse": rmse}

#     training_args = TrainingArguments(
#         output_dir="./results_deberta_gru",
#         num_train_epochs=num_epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         logging_dir="./logs",
#         logging_steps=10,
#         fp16=True,
#         load_best_model_at_end=True,
#         metric_for_best_model="rmse",
#         greater_is_better=False,
#     )

#     class CustomTrainer(Trainer):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             self.log_data = []

#         def log(self, logs, start_time=None):
#             self.log_data.append(logs)
#             super().log(logs, start_time)

#         def save_logs_to_csv(self, filename):
#             fieldnames = ["epoch", "step", "loss", "learning_rate", "eval_rmse"]
#             with open(filename, 'w', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=fieldnames)
#                 writer.writeheader()
#                 for log in self.log_data:
#                     filtered_log = {k: log.get(k, '') for k in fieldnames}
#                     writer.writerow(filtered_log)

#     trainer = CustomTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         compute_metrics=compute_metrics,
#         optimizers=(optimizer, scheduler)
#     )

#     trainer.train()
#     trainer.save_logs_to_csv("training_logs_deberta_gru.csv")
#     trainer.save_model("./finetuned_deberta_gru_best")

# if __name__ == "__main__":
#     main()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim.lr_scheduler import _LRScheduler
# from transformers import (
#     DebertaV2Tokenizer,
#     DebertaV2Model,
#     DebertaV2Config,
#     Trainer,
#     TrainingArguments
# )
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import csv
# import re


# # ================== 自定义MoE层 ==================
# class MoELayer(nn.Module):
#     def __init__(self, config, num_experts=4, k=2):
#         super().__init__()
#         self.num_experts = num_experts
#         self.k = k
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(config.hidden_size, config.intermediate_size),
#                 nn.GELU(),
#                 nn.Dropout(config.hidden_dropout_prob),
#                 nn.Linear(config.intermediate_size, config.hidden_size)
#             ) for _ in range(num_experts)
#         ])
#         self.gate = nn.Linear(config.hidden_size, num_experts)

#     def forward(self, x):
#         B, S, H = x.shape
#         x_flat = x.view(-1, H)  # [B*S, H]

#         gate_scores = self.gate(x_flat)  # [B*S, E]
#         topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=1)  # [B*S, K]
#         topk_weights = F.softmax(topk_scores, dim=1)  # [B*S, K]

#         expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # [B*S, E, H]

#         batch_indices = torch.arange(B * S).unsqueeze(1).expand(-1, self.k).to(x.device)
#         selected_expert_outputs = expert_outputs[batch_indices, topk_indices]  # [B*S, K, H]

#         output_flat = torch.einsum('bk,bkh->bh', topk_weights, selected_expert_outputs)  # [B*S, H]
#         output = output_flat.view(B, S, H)  # [B, S, H]

#         return output


# # ================== DeBERTa + MoE + GRU 模型 ==================
# class DeBERTa_MoE_Model(nn.Module):
#     def __init__(self, config, output_dim=20):
#         super().__init__()
#         self.config = config
#         self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")

#         # 替换每一层 Transformer 中的 ffn 为 MoE
#         for layer in self.deberta.encoder.layer:
#             layer.ffn = MoELayer(config).to(self.deberta.device)

#         self.number_embedding = nn.Embedding(2, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.gru = nn.GRU(
#             input_size=config.hidden_size * 2,
#             hidden_size=config.hidden_size,
#             num_layers=1,
#             batch_first=True
#         )
#         self.output_layer = nn.Linear(config.hidden_size, output_dim)

#     def forward(self, input_ids, attention_mask, number_mask, labels=None):
#         outputs = self.deberta(input_ids, attention_mask=attention_mask)
#         sequence_output = outputs.last_hidden_state  # [B, S, H]

#         cls_features = sequence_output[:, 0, :]  # [B, H]

#         num_embed = self.number_embedding(number_mask)  # [B, S, H]
#         num_features = self.dropout(num_embed.max(dim=1)[0])  # [B, H]

#         combined = torch.cat([cls_features, num_features], dim=1).unsqueeze(1)  # [B, 1, H*2]
#         gru_out, _ = self.gru(combined)  # [B, 1, H]
#         output = self.output_layer(gru_out.squeeze(1))  # [B, D]

#         if labels is not None:
#             loss = nn.functional.mse_loss(output, labels)
#             return {"loss": loss, "logits": output}
#         return {"logits": output}


# # ================== 数据集类 ==================
# class TextDataset(torch.utils.data.Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=512):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#         tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
#         number_mask = torch.zeros(self.max_length, dtype=torch.long)
#         for i, token in enumerate(tokens):
#             if re.search(r'\d', token):
#                 number_mask[i] = 1
#         return {
#             'input_ids': encoding['input_ids'].squeeze(0),
#             'attention_mask': encoding['attention_mask'].squeeze(0),
#             'number_mask': number_mask,
#             'labels': torch.tensor(label, dtype=torch.float32)
#         }


# # ================== 自定义调度器 ==================
# class CustomCosineAnnealingWarmup(_LRScheduler):
#     def __init__(self, optimizer, num_warmup_steps, num_training_steps, eta_min=0, last_epoch=-1):
#         self.num_warmup_steps = num_warmup_steps
#         self.num_training_steps = num_training_steps
#         self.eta_min = eta_min
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         current_step = self.last_epoch + 1
#         if current_step < self.num_warmup_steps:
#             return [
#                 base_lr * current_step / self.num_warmup_steps
#                 for base_lr in self.base_lrs
#             ]
#         if current_step > self.num_training_steps:
#             return [self.eta_min for _ in self.base_lrs]
#         progress = (current_step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
#         cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
#         return [
#             self.eta_min + (base_lr - self.eta_min) * cosine_decay
#             for base_lr in self.base_lrs
#         ]


# # ================== 数据预处理 ==================
# def preprocess_data():
#     with open('text_qwen.txt', 'r') as f:
#         text = f.readlines()
#     with open('data-augmented2.txt', 'r') as f:
#         labels = [list(map(float, line.strip().split(',')))[12:] for line in f if line.strip()]
    
#     assert len(text) == len(labels), "文本和标签数量不一致"

#     text = text[:116000]
#     labels = labels[:116000]

#     scaler = StandardScaler()
#     labels = scaler.fit_transform(np.array(labels)).tolist()
#     return train_test_split(text, labels, test_size=0.2, random_state=42), scaler


# # ================== 计算指标 ==================
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = scaler.inverse_transform(predictions)
#     labels = scaler.inverse_transform(labels)
#     rmse = np.sqrt(mean_squared_error(labels, predictions))
#     return {"rmse": rmse}


# # ================== 自定义 Trainer 类 ==================
# class CustomTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.log_data = []

#     def log(self, logs, start_time=None):
#         self.log_data.append(logs)
#         super().log(logs, start_time)

#     def save_logs_to_csv(self, filename):
#         fieldnames = ["epoch", "step", "loss", "learning_rate", "eval_rmse"]
#         with open(filename, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             for log in self.log_data:
#                 filtered_log = {k: log.get(k, '') for k in fieldnames}
#                 writer.writerow(filtered_log)


# # ================== 主函数 ==================
# def main():
#     config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-base")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")

#     (train_texts, eval_texts, train_labels, eval_labels), scaler = preprocess_data()

#     tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
#     train_dataset = TextDataset(train_texts, train_labels, tokenizer)
#     eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer)

#     model = DeBERTa_MoE_Model(config, output_dim=20).to(device)

#     num_epochs = 10
#     dataset_size = len(train_dataset)
#     batch_size = 32
#     total_steps = (num_epochs * dataset_size) // batch_size
#     optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, eps=1e-6)
#     scheduler = CustomCosineAnnealingWarmup(optimizer, 3000, total_steps, eta_min=1e-7)

#     training_args = TrainingArguments(
#         output_dir="./results_deberta_gru",
#         num_train_epochs=num_epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         logging_dir="./logs",
#         logging_steps=10,
#         fp16=True,
#         load_best_model_at_end=True,
#         metric_for_best_model="rmse",
#         greater_is_better=False,
#         report_to='none'
#     )

#     trainer = CustomTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         compute_metrics=compute_metrics,
#         optimizers=(optimizer, scheduler)
#     )

#     trainer.train()
#     trainer.save_logs_to_csv("training_logs_deberta_gru2.csv")
#     trainer.save_model("./finetuned_deberta_gru_best2")


# if __name__ == "__main__":
#     main()

# 进行修改过后的deberta_gru_moe
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2Model,
    DebertaV2Config,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import csv
import re

# ================== 全局变量定义 ==================
global_scaler = None  # 用于在 compute_metrics 中访问 scaler


# ================== 自定义MoE层 ==================
class MoELayer(nn.Module):
    def __init__(self, config, num_experts=6, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.intermediate_size, config.hidden_size)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(config.hidden_size, num_experts)

    def forward(self, x):
        B, S, H = x.shape
        x_flat = x.view(-1, H)  # [B*S, H]

        gate_scores = self.gate(x_flat)  # [B*S, E]
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=1)  # [B*S, K]
        topk_weights = F.softmax(topk_scores, dim=1)  # [B*S, K]

        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # [B*S, E, H]

        batch_indices = torch.arange(B * S).unsqueeze(1).expand(-1, self.k).to(x.device)
        selected_expert_outputs = expert_outputs[batch_indices, topk_indices]  # [B*S, K, H]

        output_flat = torch.einsum('bk,bkh->bh', topk_weights, selected_expert_outputs)  # [B*S, H]
        output = output_flat.view(B, S, H)  # [B, S, H]

        return output


# ================== PositionalEncoding 类 ==================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, 1, D]
        return x + self.pe[:x.size(1), :]


# ================== DeBERTa + MoE + GRU 模型 ==================
class DeBERTa_MoE_Model(nn.Module):
    def __init__(self, config, output_dim=20):
        super().__init__()
        self.config = config
        self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")

        # 替换每一层 Transformer 中的 ffn 为 MoE
        for layer in self.deberta.encoder.layer:
            layer.ffn = MoELayer(config).to(self.deberta.device)

        self.number_embedding = nn.Embedding(2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gru = nn.GRU(
            input_size=config.hidden_size * 2,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.ln = nn.LayerNorm(config.hidden_size)  # LayerNorm
        self.output_layer = nn.Linear(config.hidden_size, output_dim)

        # 可学习位置编码
        self.pos_encoder = PositionalEncoding(config.hidden_size * 2)

    def forward(self, input_ids, attention_mask, number_mask, labels=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, S, H]

        cls_features = sequence_output[:, 0, :]  # [B, H]

        num_embed = self.number_embedding(number_mask)  # [B, S, H]
        num_features = self.dropout(num_embed.max(dim=1)[0])  # [B, H]

        combined = torch.cat([cls_features, num_features], dim=1).unsqueeze(1)  # [B, 1, H*2]
        combined = self.pos_encoder(combined)  # 添加位置编码

        gru_out, _ = self.gru(combined)  # [B, 1, H]
        gru_out = self.ln(gru_out)  # LayerNorm
        output = self.output_layer(gru_out.squeeze(1))  # [B, D]

        if labels is not None:
            loss = nn.functional.mse_loss(output, labels)
            return {"loss": loss, "logits": output}
        return {"logits": output}


# ================== 数据集类 ==================
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        number_mask = torch.zeros(self.max_length, dtype=torch.long)
        for i, token in enumerate(tokens):
            if re.search(r'\d', token):
                number_mask[i] = 1
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'number_mask': number_mask,
            'labels': torch.tensor(label, dtype=torch.float32)
        }


# ================== 自定义调度器 ==================
class CustomCosineAnnealingWarmup(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, eta_min=0, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1
        if current_step < self.num_warmup_steps:
            return [
                base_lr * current_step / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]
        if current_step > self.num_training_steps:
            return [self.eta_min for _ in self.base_lrs]
        progress = (current_step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]


# ================== 数据预处理 ==================
def preprocess_data():
    global global_scaler  # 声明使用全局变量
    with open('text_qwen.txt', 'r') as f:
        text = f.readlines()
    with open('data-augmented2.txt', 'r') as f:
        labels = [list(map(float, line.strip().split(',')))[12:] for line in f if line.strip()]

    assert len(text) == len(labels), "文本和标签数量不一致"

    text = text[:116000]
    labels = labels[:116000]

    scaler = StandardScaler()
    labels = scaler.fit_transform(np.array(labels)).tolist()
    global_scaler = scaler  # 存入全局变量

    return train_test_split(text, labels, test_size=0.2, random_state=42), scaler


# ================== 计算指标 ==================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = global_scaler.inverse_transform(predictions)
    labels = global_scaler.inverse_transform(labels)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return {"rmse": rmse}


# ================== 自定义 Trainer 类 ==================
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_data = []

    def log(self, logs, start_time=None):
        self.log_data.append(logs)
        super().log(logs, start_time)

    def save_logs_to_csv(self, filename):
        fieldnames = ["epoch", "step", "loss", "learning_rate", "eval_rmse"]
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for log in self.log_data:
                filtered_log = {k: log.get(k, '') for k in fieldnames}
                writer.writerow(filtered_log)


# ================== 主函数 ==================
def main():
    config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    (train_texts, eval_texts, train_labels, eval_labels), scaler = preprocess_data()

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer)

    model = DeBERTa_MoE_Model(config, output_dim=20).to(device)

    num_epochs = 10
    dataset_size = len(train_dataset)
    batch_size = 32
    total_steps = (num_epochs * dataset_size) // batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, eps=1e-6)
    scheduler = CustomCosineAnnealingWarmup(optimizer, 3000, total_steps, eta_min=1e-7)

    training_args = TrainingArguments(
        output_dir="./results_deberta_gru",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        report_to='none'
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()
    trainer.save_logs_to_csv("training_logs_deberta_gru_8 expert.csv")
    trainer.save_model("./finetuned_deberta_gru_8 expert")


if __name__ == "__main__":
    main()


# 没有moe的deberta-gru

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import _LRScheduler
# from transformers import (
#     DebertaV2Tokenizer,
#     DebertaV2Model,
#     DebertaV2Config,
#     Trainer,
#     TrainingArguments
# )
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import csv
# import re

# # ================== 全局变量定义 ==================
# global_scaler = None  # 用于在 compute_metrics 中访问 scaler


# # ================== PositionalEncoding 类 ==================
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x: [B, 1, D]
#         return x + self.pe[:x.size(1), :]


# # ================== DeBERTa + GRU 模型（无 MoE）==================
# class DeBERTa_MoE_Model(nn.Module):
#     def __init__(self, config, output_dim=20):
#         super().__init__()
#         self.config = config
#         self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")

#         # 不再替换为 MoE，使用原始 DeBERTa 的 ffn 层

#         self.number_embedding = nn.Embedding(2, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.gru = nn.GRU(
#             input_size=config.hidden_size * 2,
#             hidden_size=config.hidden_size,
#             num_layers=1,
#             batch_first=True
#         )
#         self.ln = nn.LayerNorm(config.hidden_size)  # LayerNorm
#         self.output_layer = nn.Linear(config.hidden_size, output_dim)

#         # 可学习位置编码
#         self.pos_encoder = PositionalEncoding(config.hidden_size * 2)

#     def forward(self, input_ids, attention_mask, number_mask, labels=None):
#         outputs = self.deberta(input_ids, attention_mask=attention_mask)
#         sequence_output = outputs.last_hidden_state  # [B, S, H]

#         cls_features = sequence_output[:, 0, :]  # [B, H]

#         num_embed = self.number_embedding(number_mask)  # [B, S, H]
#         num_features = self.dropout(num_embed.max(dim=1)[0])  # [B, H]

#         combined = torch.cat([cls_features, num_features], dim=1).unsqueeze(1)  # [B, 1, H*2]
#         combined = self.pos_encoder(combined)  # 添加位置编码

#         gru_out, _ = self.gru(combined)  # [B, 1, H]
#         gru_out = self.ln(gru_out)  # LayerNorm
#         output = self.output_layer(gru_out.squeeze(1))  # [B, D]

#         if labels is not None:
#             loss = nn.functional.mse_loss(output, labels)
#             return {"loss": loss, "logits": output}
#         return {"logits": output}


# # ================== 数据集类 ==================
# class TextDataset(torch.utils.data.Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=512):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#         tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
#         number_mask = torch.zeros(self.max_length, dtype=torch.long)
#         for i, token in enumerate(tokens):
#             if re.search(r'\d', token):
#                 number_mask[i] = 1
#         return {
#             'input_ids': encoding['input_ids'].squeeze(0),
#             'attention_mask': encoding['attention_mask'].squeeze(0),
#             'number_mask': number_mask,
#             'labels': torch.tensor(label, dtype=torch.float32)
#         }


# # ================== 自定义调度器 ==================
# class CustomCosineAnnealingWarmup(_LRScheduler):
#     def __init__(self, optimizer, num_warmup_steps, num_training_steps, eta_min=0, last_epoch=-1):
#         self.num_warmup_steps = num_warmup_steps
#         self.num_training_steps = num_training_steps
#         self.eta_min = eta_min
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         current_step = self.last_epoch + 1
#         if current_step < self.num_warmup_steps:
#             return [
#                 base_lr * current_step / self.num_warmup_steps
#                 for base_lr in self.base_lrs
#             ]
#         if current_step > self.num_training_steps:
#             return [self.eta_min for _ in self.base_lrs]
#         progress = (current_step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
#         cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
#         return [
#             self.eta_min + (base_lr - self.eta_min) * cosine_decay
#             for base_lr in self.base_lrs
#         ]


# # ================== 数据预处理 ==================
# def preprocess_data():
#     global global_scaler  # 声明使用全局变量
#     with open('text_qwen.txt', 'r') as f:
#         text = f.readlines()
#     with open('data-augmented2.txt', 'r') as f:
#         labels = [list(map(float, line.strip().split(',')))[12:] for line in f if line.strip()]

#     assert len(text) == len(labels), "文本和标签数量不一致"

#     text = text[:116000]
#     labels = labels[:116000]

#     scaler = StandardScaler()
#     labels = scaler.fit_transform(np.array(labels)).tolist()
#     global_scaler = scaler  # 存入全局变量

#     return train_test_split(text, labels, test_size=0.2, random_state=42), scaler


# # ================== 计算指标 ==================
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = global_scaler.inverse_transform(predictions)
#     labels = global_scaler.inverse_transform(labels)
#     rmse = np.sqrt(mean_squared_error(labels, predictions))
#     return {"rmse": rmse}


# # ================== 自定义 Trainer 类 ==================
# class CustomTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.log_data = []

#     def log(self, logs, start_time=None):
#         self.log_data.append(logs)
#         super().log(logs, start_time)

#     def save_logs_to_csv(self, filename):
#         fieldnames = ["epoch", "step", "loss", "learning_rate", "eval_rmse"]
#         with open(filename, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             for log in self.log_data:
#                 filtered_log = {k: log.get(k, '') for k in fieldnames}
#                 writer.writerow(filtered_log)


# # ================== 主函数 ==================
# def main():
#     config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-base")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")

#     (train_texts, eval_texts, train_labels, eval_labels), scaler = preprocess_data()

#     tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
#     train_dataset = TextDataset(train_texts, train_labels, tokenizer)
#     eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer)

#     model = DeBERTa_MoE_Model(config, output_dim=20).to(device)

#     num_epochs = 10
#     dataset_size = len(train_dataset)
#     batch_size = 32
#     total_steps = (num_epochs * dataset_size) // batch_size
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, eps=1e-6)
#     scheduler = CustomCosineAnnealingWarmup(optimizer, 3000, total_steps, eta_min=1e-7)

#     training_args = TrainingArguments(
#         output_dir="./results_deberta_gru",
#         num_train_epochs=num_epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         logging_dir="./logs",
#         logging_steps=10,
#         fp16=True,
#         load_best_model_at_end=True,
#         metric_for_best_model="rmse",
#         greater_is_better=False,
#         report_to='none'
#     )

#     trainer = CustomTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         compute_metrics=compute_metrics,
#         optimizers=(optimizer, scheduler)
#     )

#     trainer.train()
#     trainer.save_logs_to_csv("training_logs_deberta_gru_without moe.csv")
#     trainer.save_model("./finetuned_deberta_gru_best_without moe")


# if __name__ == "__main__":
#     main()
