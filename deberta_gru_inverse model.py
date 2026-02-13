# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2Config
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import pandas as pd
# import time, re

# class TextDataset(Dataset):
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


# class DeBERTa_GRU_Model(nn.Module):
#     def __init__(self, deberta_config, gru_hidden_dims=[256, 512], num_classes=3, num_regression_targets=11, num_experts=4):
#         super().__init__()
#         self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
#         self.number_embedding = nn.Embedding(2, deberta_config.hidden_size)
#         self.dropout = nn.Dropout(deberta_config.hidden_dropout_prob)

#         self.gru_experts = nn.ModuleList([
#             GRUExpert(input_dim=deberta_config.hidden_size * 2, hidden_dims=gru_hidden_dims,
#                       num_classes=num_classes, num_regression_targets=num_regression_targets)
#             for _ in range(num_experts)
#         ])
#         self.gate = SequenceGatingNetwork(input_dim=deberta_config.hidden_size * 2, hidden_dim=128, num_experts=num_experts)

#     def forward(self, input_ids, attention_mask, number_mask, labels=None):
#         outputs = self.deberta(input_ids, attention_mask=attention_mask)
#         sequence_output = outputs.last_hidden_state

#         cls_features = sequence_output[:, 0, :]
#         num_embed = self.number_embedding(number_mask)
#         num_features = self.dropout(num_embed.max(dim=1)[0])

#         combined = torch.cat([cls_features, num_features], dim=1).unsqueeze(1)

#         class_outputs, regress_outputs = zip(*[expert(combined) for expert in self.gru_experts])
#         class_outputs = torch.stack(class_outputs, dim=1)
#         regress_outputs = torch.stack(regress_outputs, dim=1)
#         gate_weights = self.gate(combined)
#         class_output = torch.sum(class_outputs * gate_weights.unsqueeze(-1), dim=1)
#         regress_output = torch.sum(regress_outputs * gate_weights.unsqueeze(-1), dim=1)

#         if labels is not None:
#             y_class = labels[:, 0].long()
#             y_regress = labels[:, 1:]
#             loss_class = nn.CrossEntropyLoss()(class_output, y_class)
#             loss_regress = compute_weighted_mse_loss(regress_output, y_regress, weights=self.get_regression_weights(y_regress))
#             loss = loss_class + loss_regress
#             return {"loss": loss, "logits": (class_output, regress_output)}
#         return {"logits": (class_output, regress_output)}

#     def get_regression_weights(self, y_regress):
#         regression_std = torch.std(y_regress, dim=0)
#         return torch.tensor([1 / std for std in regression_std], device=y_regress.device)

# class GRUExpert(nn.Module):
#     def __init__(self, input_dim, hidden_dims, num_classes, num_regression_targets):
#         super().__init__()
#         self.hidden_dims = hidden_dims
#         self.num_layers = len(hidden_dims)
#         self.gru_cells = nn.ModuleList(
#             [nn.GRUCell(input_dim, hidden_dims[0])] +
#             [nn.GRUCell(hidden_dims[i-1], hidden_dims[i]) for i in range(1, self.num_layers)]
#         )
#         self.classifier = nn.Linear(hidden_dims[-1], num_classes)
#         self.regressor = nn.Linear(hidden_dims[-1], num_regression_targets)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#         h = [torch.zeros(batch_size, dim, device=x.device) for dim in self.hidden_dims]
#         for t in range(seq_len):
#             h[0] = self.gru_cells[0](x[:, t, :], h[0])
#             for i in range(1, self.num_layers):
#                 h[i] = self.gru_cells[i](h[i-1], h[i])
#         features = h[-1]

#         class_out = self.classifier(features)
#         raw_regress_out = self.regressor(features)

#         regress_out = torch.zeros_like(raw_regress_out)
#         regress_out[:, 0] = torch.sigmoid(raw_regress_out[:, 0]) * 8
#         regress_out[:, 1:4] = torch.sigmoid(raw_regress_out[:, 1:4]) * 0.2 + 0.05
#         regress_out[:, 4:7] = torch.sigmoid(raw_regress_out[:, 4:7]) * 0.5
#         regress_out[:, 7] = torch.sigmoid(raw_regress_out[:, 7]) * 10 + 10
#         regress_out[:, 8:11] = torch.sigmoid(raw_regress_out[:, 8:11]) * 2 + 1

#         return class_out, regress_out

# class SequenceGatingNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_experts):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, num_experts)
#         )

#     def forward(self, x):
#         pooled = x.mean(dim=1)
#         return torch.softmax(self.net(pooled), dim=1)

# def compute_weighted_mse_loss(pred, target, weights):
#     squared_error = (pred - target) ** 2
#     weighted_squared_error = squared_error * weights.unsqueeze(0)
#     return torch.mean(weighted_squared_error)

# def load_data():
#     data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
#     y = data.iloc[:116000, :12].values
#     X_texts = data.iloc[:116000, 12:].astype(str).apply(lambda row: ','.join(row), axis=1).tolist()

#     scaler_y = StandardScaler()
#     y_reg = scaler_y.fit_transform(y[:, 1:])
#     y = np.hstack([y[:, 0].reshape(-1, 1), y_reg])

#     return train_test_split(X_texts, y, test_size=0.2, random_state=42), scaler_y

# def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=10, device="cuda"):
#     model.to(device)
#     best_val_loss = float('inf')
#     best_model_state = None

#     total_batches = len(train_loader)
#     progress_interval = 0.05  
#     batches_per_progress = int(total_batches * progress_interval)

#     for epoch in range(epochs):
#         start_time = time.time()
#         model.train()
#         epoch_loss = 0.0

#         for i, batch in enumerate(train_loader):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             number_mask = batch['number_mask'].to(device)
#             labels = batch['labels'].to(device)

#             optimizer.zero_grad()
#             outputs = model(input_ids, attention_mask, number_mask, labels=labels)
#             loss = outputs["loss"]
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item() * input_ids.size(0)

#             if (i + 1) % batches_per_progress == 0 or (i + 1) == total_batches:
#                 current_progress = (i + 1) / total_batches
#                 avg_loss_so_far = epoch_loss / ((i + 1) * train_loader.batch_size)
#                 print(f"Epoch [{epoch+1}/{epochs}] - Progress: {current_progress:.2f} "
#                       f"- Avg Loss: {avg_loss_so_far:.6f}")

#         avg_train_loss = epoch_loss / len(train_loader.dataset)

#         model.eval()
#         val_loss = 0.0
#         total_samples = 0

#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 number_mask = batch['number_mask'].to(device)
#                 labels = batch['labels'].to(device)

#                 outputs = model(input_ids, attention_mask, number_mask, labels=labels)
#                 loss = outputs["loss"]

#                 val_loss += loss.item() * input_ids.size(0)
#                 total_samples += input_ids.size(0)

#         val_loss /= total_samples
#         scheduler.step(val_loss)

#         end_time = time.time()
#         epoch_duration = end_time - start_time

#         print(f"Epoch [{epoch+1}/{epochs}] "
#               f"- Train Loss: {avg_train_loss:.6f} "
#               f"- Val Loss: {val_loss:.6f} "
#               f"- LR: {optimizer.param_groups[0]['lr']:.6f} "
#               f"- Time: {epoch_duration:.2f}s")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_state = model.state_dict().copy()

#     model.load_state_dict(best_model_state)
#     return best_val_loss

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     (train_texts, eval_texts, train_labels, eval_labels), scaler_y = load_data()

#     tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
#     train_dataset = TextDataset(train_texts, train_labels, tokenizer)
#     eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer)

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

#     deberta_config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-base")
#     model = DeBERTa_GRU_Model(deberta_config).to(device)

#     optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

#     train_model(model, train_loader, eval_loader, optimizer, scheduler, epochs=20, device=device)

#     torch.save(model.state_dict(), "merged_deberta_gru_model.pt")

# if __name__ == '__main__':
#     main()

