# # 没有使用sigmoid
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import os
# import time
# import numpy as np
# from torch.utils.data import DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # ========================
# # 数据加载 & 预处理
# # ========================
# def load_data():
#     data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
#     y = data.iloc[:, :12].values
#     X = data.iloc[:, 12:].values
#     scaler_x = StandardScaler()
#     X = scaler_x.fit_transform(X)
#     scaler_y = StandardScaler()
#     y[:, 1:] = scaler_y.fit_transform(y[:, 1:])  # 只对回归部分标准化
#     y[:, 0] = y[:, 0].astype(int)  # 分类标签
#     num_samples = X.shape[0]
#     X = X.reshape(num_samples, 20, 1)
#     return X, y, scaler_x, scaler_y

# # ========================
# # GRU Expert
# # ========================
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
#         return self.classifier(h[-1]), self.regressor(h[-1])

# # ========================
# # Gating Network
# # ========================
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

# # ========================
# # MoE-GRU Network
# # ========================
# class GRUMoENetwork(nn.Module):
#     def __init__(self, input_dim=1, hidden_dims=[128, 256, 512],
#                  num_classes=3, num_regression_targets=11, num_experts=4):
#         super().__init__()
#         self.experts = nn.ModuleList([
#             GRUExpert(input_dim, hidden_dims, num_classes, num_regression_targets)
#             for _ in range(num_experts)
#         ])
#         self.gate = SequenceGatingNetwork(input_dim=input_dim, hidden_dim=128, num_experts=num_experts)

#     def forward(self, x):
#         class_outputs, regress_outputs = zip(*[expert(x) for expert in self.experts])
#         class_outputs = torch.stack(class_outputs, dim=1)
#         regress_outputs = torch.stack(regress_outputs, dim=1)
#         gate_weights = self.gate(x)
#         class_output = torch.sum(class_outputs * gate_weights.unsqueeze(-1), dim=1)
#         regress_output = torch.sum(regress_outputs * gate_weights.unsqueeze(-1), dim=1)
#         return class_output, regress_output, gate_weights

# # ========================
# # 训练函数
# # ========================
# def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=10, device="cuda", scaler_y=None):
#     criterion_class = nn.CrossEntropyLoss()
#     criterion_regress = nn.MSELoss()

#     model.to(device)
#     best_val_loss = float('inf')
#     best_model_state = None

#     for epoch in range(epochs):
#         start_time = time.time()
#         model.train()
#         epoch_loss = 0.0

#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)
#             y_class = y_batch[:, 0].long()
#             y_regress = y_batch[:, 1:]

#             optimizer.zero_grad()
#             out_class, out_regress, _ = model(X_batch)
#             loss_class = criterion_class(out_class, y_class)
#             loss_regress = criterion_regress(out_regress, y_regress)
#             loss = loss_class + loss_regress
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item() * X_batch.size(0)

#         avg_train_loss = epoch_loss / len(train_loader.dataset)

#         # --------------------------
#         # 验证阶段
#         # --------------------------
#         model.eval()
#         val_loss = 0.0
#         val_mae = 0.0
#         total_samples = 0

#         with torch.no_grad():
#             for X_val, y_val in val_loader:
#                 X_val = X_val.to(device)
#                 y_val = y_val.to(device)
#                 y_class = y_val[:, 0].long()
#                 y_regress = y_val[:, 1:]

#                 out_class, out_regress, _ = model(X_val)
#                 loss_class = criterion_class(out_class, y_class)
#                 loss_regress = criterion_regress(out_regress, y_regress)
#                 loss = loss_class + loss_regress

#                 val_loss += loss.item() * X_val.size(0)
#                 total_samples += X_val.size(0)

#                 # 反标准化后的 MAE
#                 if scaler_y is not None:
#                     y_pred = scaler_y.inverse_transform(out_regress.cpu().numpy())
#                     y_true = scaler_y.inverse_transform(y_regress.cpu().numpy())
#                     val_mae += np.sum(np.abs(y_pred - y_true))

#         val_loss /= total_samples
#         val_mae /= total_samples
#         scheduler.step(val_loss)

#         end_time = time.time()
#         epoch_duration = end_time - start_time

#         print(f"Epoch [{epoch+1}/{epochs}] "
#               f"- Train Loss: {avg_train_loss:.6f} "
#               f"- Val Loss: {val_loss:.6f} "
#               f"- Val MAE: {val_mae:.6f} "
#               f"- LR: {optimizer.param_groups[0]['lr']:.6f} "
#               f"- Time: {epoch_duration:.2f}s")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_state = model.state_dict().copy()

#     model.load_state_dict(best_model_state)
#     return best_val_loss

# # ========================
# # 训练入口
# # ========================
# def train_once():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     X, y, scaler_x, scaler_y = load_data()
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     train_loader = DataLoader([
#         (torch.tensor(X_train[i], dtype=torch.float32),
#          torch.tensor(y_train[i], dtype=torch.float32)) for i in range(len(X_train))
#     ], batch_size=512, shuffle=True)

#     val_loader = DataLoader([
#         (torch.tensor(X_val[i], dtype=torch.float32),
#          torch.tensor(y_val[i], dtype=torch.float32)) for i in range(len(X_val))
#     ], batch_size=512, shuffle=False)

#     model = GRUMoENetwork(
#         hidden_dims=[256, 512, 1024, 2048],
#         num_experts=4,
#         num_classes=3,
#         num_regression_targets=11
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
#     )

#     train_model(model, train_loader, val_loader, optimizer, scheduler,
#                 epochs=800, device=device, scaler_y=scaler_y)

#     torch.save(model.state_dict(), "models/best_moe_gru_without sigmoid.pt")
#     print("模型已保存至 models/best_moe_multitask_model.pt")

# # ========================
# # 执行主训练
# # ========================
# if __name__ == '__main__':
#     os.makedirs("models", exist_ok=True)
#     train_once()


# #没有考虑回归权重的模型
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import os
# import time
# import numpy as np
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # ========================
# # 数据加载 & 预处理
# # ========================
# def load_data():
#     data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
#     y = data.iloc[:, :12].values
#     X = data.iloc[:, 12:].values

#     # 输入标准化
#     scaler_x = StandardScaler()
#     X = scaler_x.fit_transform(X)

#     y[:, 0] = y[:, 0].astype(int)  # 分类标签，其他回归部分不再标准化
#     num_samples = X.shape[0]
#     X = X.reshape(num_samples, 20, 1)
#     return X, y, scaler_x, None  # 不返回 scaler_y

# # ========================
# # GRU Expert
# # ========================
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

#         # 映射到指定范围
#         regress_out = torch.zeros_like(raw_regress_out)
#         regress_out[:, 0] = torch.sigmoid(raw_regress_out[:, 0]) * 8
#         regress_out[:, 1:4] = torch.sigmoid(raw_regress_out[:, 1:4]) * 0.2 + 0.05
#         regress_out[:, 4:7] = torch.sigmoid(raw_regress_out[:, 4:7]) * 0.5
#         regress_out[:, 7] = torch.sigmoid(raw_regress_out[:, 7]) * 10 + 10
#         regress_out[:, 8:11] = torch.sigmoid(raw_regress_out[:, 8:11]) * 2 + 1

#         return class_out, regress_out

# # ========================
# # Gating Network
# # ========================
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

# # ========================
# # MoE-GRU Network
# # ========================
# class GRUMoENetwork(nn.Module):
#     def __init__(self, input_dim=1, hidden_dims=[ 256, 512],
#                  num_classes=3, num_regression_targets=11, num_experts=4):
#         super().__init__()
#         self.experts = nn.ModuleList([
#             GRUExpert(input_dim, hidden_dims, num_classes, num_regression_targets)
#             for _ in range(num_experts)
#         ])
#         self.gate = SequenceGatingNetwork(input_dim=input_dim, hidden_dim=128, num_experts=num_experts)

#     def forward(self, x):
#         class_outputs, regress_outputs = zip(*[expert(x) for expert in self.experts])
#         class_outputs = torch.stack(class_outputs, dim=1)
#         regress_outputs = torch.stack(regress_outputs, dim=1)
#         gate_weights = self.gate(x)
#         class_output = torch.sum(class_outputs * gate_weights.unsqueeze(-1), dim=1)
#         regress_output = torch.sum(regress_outputs * gate_weights.unsqueeze(-1), dim=1)
#         return class_output, regress_output, gate_weights

# # ========================
# # 训练函数
# # ========================
# def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=10, device="cuda", scaler_y=None):
#     criterion_class = nn.CrossEntropyLoss()
#     criterion_regress = nn.MSELoss()

#     model.to(device)
#     best_val_loss = float('inf')
#     best_model_state = None

#     for epoch in range(epochs):
#         start_time = time.time()
#         model.train()
#         epoch_loss = 0.0

#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)
#             y_class = y_batch[:, 0].long()
#             y_regress = y_batch[:, 1:]

#             optimizer.zero_grad()
#             out_class, out_regress, _ = model(X_batch)
#             loss_class = criterion_class(out_class, y_class)
#             loss_regress = criterion_regress(out_regress, y_regress)
#             loss = loss_class + loss_regress
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item() * X_batch.size(0)

#         avg_train_loss = epoch_loss / len(train_loader.dataset)

#         # 验证阶段
#         model.eval()
#         val_loss = 0.0
#         val_mae = 0.0
#         total_samples = 0

#         with torch.no_grad():
#             for X_val, y_val in val_loader:
#                 X_val = X_val.to(device)
#                 y_val = y_val.to(device)
#                 y_class = y_val[:, 0].long()
#                 y_regress = y_val[:, 1:]

#                 out_class, out_regress, _ = model(X_val)
#                 loss_class = criterion_class(out_class, y_class)
#                 loss_regress = criterion_regress(out_regress, y_regress)
#                 loss = loss_class + loss_regress

#                 val_loss += loss.item() * X_val.size(0)
#                 total_samples += X_val.size(0)

#                 # 直接计算 MAE
#                 val_mae += torch.sum(torch.abs(out_regress - y_regress)).item()

#         val_loss /= total_samples
#         val_mae /= total_samples
#         scheduler.step(val_loss)

#         end_time = time.time()
#         epoch_duration = end_time - start_time

#         print(f"Epoch [{epoch+1}/{epochs}] "
#               f"- Train Loss: {avg_train_loss:.6f} "
#               f"- Val Loss: {val_loss:.6f} "
#               f"- Val MAE: {val_mae:.6f} "
#               f"- LR: {optimizer.param_groups[0]['lr']:.6f} "
#               f"- Time: {epoch_duration:.2f}s")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_state = model.state_dict().copy()

#     model.load_state_dict(best_model_state)
#     return best_val_loss

# # ========================
# # 训练入口
# # ========================
# def train_once():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     X, y, scaler_x, _ = load_data()
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     train_loader = DataLoader([
#         (torch.tensor(X_train[i], dtype=torch.float32),
#          torch.tensor(y_train[i], dtype=torch.float32)) for i in range(len(X_train))
#     ], batch_size=512, shuffle=True)

#     val_loader = DataLoader([
#         (torch.tensor(X_val[i], dtype=torch.float32),
#          torch.tensor(y_val[i], dtype=torch.float32)) for i in range(len(X_val))
#     ], batch_size=512, shuffle=False)

#     model = GRUMoENetwork(
#         hidden_dims=[256, 512, 1024, 2048],
#         num_experts=4,
#         num_classes=3,
#         num_regression_targets=11
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=25, min_lr=1e-6
#     )

#     train_model(model, train_loader, val_loader, optimizer, scheduler,
#                 epochs=800, device=device, scaler_y=None)

#     torch.save(model.state_dict(), "models/best_moe_gru_without regression.pt")
#     print("模型已保存至 models/best_moe_multitask_model.pt")

# # ========================
# # 执行主训练
# # ========================
# if __name__ == '__main__':
#     os.makedirs("models", exist_ok=True)
#     train_once()

# # 考虑回归权重的模型
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import os
# import time
# import numpy as np
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# def load_data():
#     data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
#     y = data.iloc[:, :12].values
#     X = data.iloc[:, 12:].values

#     scaler_x = StandardScaler()
#     X = scaler_x.fit_transform(X)

#     y[:, 0] = y[:, 0].astype(int) 
#     num_samples = X.shape[0]
#     X = X.reshape(num_samples, 20, 1)
#     return X, y, scaler_x, None 

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

# class GRUMoENetwork(nn.Module):
#     def __init__(self, input_dim=1, hidden_dims=[256, 512],
#                  num_classes=3, num_regression_targets=11, num_experts=4):
#         super().__init__()
#         self.experts = nn.ModuleList([
#             GRUExpert(input_dim, hidden_dims, num_classes, num_regression_targets)
#             for _ in range(num_experts)
#         ])
#         self.gate = SequenceGatingNetwork(input_dim=input_dim, hidden_dim=128, num_experts=num_experts)

#     def forward(self, x):
#         class_outputs, regress_outputs = zip(*[expert(x) for expert in self.experts])
#         class_outputs = torch.stack(class_outputs, dim=1)
#         regress_outputs = torch.stack(regress_outputs, dim=1)
#         gate_weights = self.gate(x)
#         class_output = torch.sum(class_outputs * gate_weights.unsqueeze(-1), dim=1)
#         regress_output = torch.sum(regress_outputs * gate_weights.unsqueeze(-1), dim=1)
#         return class_output, regress_output, gate_weights

# def compute_weighted_mse_loss(pred, target, weights):
#     squared_error = (pred - target) ** 2
#     weighted_squared_error = squared_error * weights.unsqueeze(0)
#     return torch.mean(weighted_squared_error)

# def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=10, device="cuda", weights=None):
#     criterion_class = nn.CrossEntropyLoss()

#     model.to(device)
#     best_val_loss = float('inf')
#     best_model_state = None

#     for epoch in range(epochs):
#         start_time = time.time()
#         model.train()
#         epoch_loss = 0.0

#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)
#             y_class = y_batch[:, 0].long()
#             y_regress = y_batch[:, 1:]

#             optimizer.zero_grad()
#             out_class, out_regress, _ = model(X_batch)

#             print("out_class shape:", out_class.shape)
#             print("y_class:", y_class)
#             assert torch.all((y_class >= 0) & (y_class < out_class.size(1))), "y_class 中存在非法标签！"

#             loss_class = criterion_class(out_class, y_class)
#             loss_regress = compute_weighted_mse_loss(out_regress, y_regress, weights)
#             loss = loss_class + loss_regress
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item() * X_batch.size(0)

#         avg_train_loss = epoch_loss / len(train_loader.dataset)

#         # 验证阶段
#         model.eval()
#         val_loss = 0.0
#         val_mae = 0.0
#         total_samples = 0

#         with torch.no_grad():
#             for X_val, y_val in val_loader:
#                 X_val = X_val.to(device)
#                 y_val = y_val.to(device)
#                 y_class = y_val[:, 0].long()
#                 y_regress = y_val[:, 1:]

#                 out_class, out_regress, _ = model(X_val)

#                 print("out_class shape:", out_class.shape)
#                 print("y_class:", y_class)
#                 assert torch.all((y_class >= 0) & (y_class < out_class.size(1))), "y_class 中存在非法标签！"

#                 loss_class = criterion_class(out_class, y_class)
#                 loss_regress = compute_weighted_mse_loss(out_regress, y_regress, weights)
#                 loss = loss_class + loss_regress

#                 val_loss += loss.item() * X_val.size(0)
#                 total_samples += X_val.size(0)

#                 val_mae += torch.sum(torch.abs(out_regress - y_regress)).item()

#         val_loss /= total_samples
#         val_mae /= total_samples
#         scheduler.step(val_loss)

#         end_time = time.time()
#         epoch_duration = end_time - start_time

#         print(f"Epoch [{epoch+1}/{epochs}] "
#               f"- Train Loss: {avg_train_loss:.6f} "
#               f"- Val Loss: {val_loss:.6f} "
#               f"- Val MAE: {val_mae:.6f} "
#               f"- LR: {optimizer.param_groups[0]['lr']:.6f} "
#               f"- Time: {epoch_duration:.2f}s")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_state = model.state_dict().copy()

#     model.load_state_dict(best_model_state)
#     return best_val_loss

# def train_once():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     X, y, scaler_x, _ = load_data()
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     train_loader = DataLoader([
#         (torch.tensor(X_train[i], dtype=torch.float32),
#          torch.tensor(y_train[i], dtype=torch.float32)) for i in range(len(X_train))
#     ], batch_size=512, shuffle=True)

#     val_loader = DataLoader([
#         (torch.tensor(X_val[i], dtype=torch.float32),
#          torch.tensor(y_val[i], dtype=torch.float32)) for i in range(len(X_val))
#     ], batch_size=512, shuffle=False)

#     regression_std = np.std(y_train[:, 1:], axis=0)
#     weights = torch.tensor([1 / std for std in regression_std], device=device)

#     model = GRUMoENetwork(
#         hidden_dims=[256, 512, 1024, 2048],
#         num_experts=4,
#         num_classes=3,
#         num_regression_targets=11
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=25, min_lr=1e-6
#     )

#     train_model(model, train_loader, val_loader, optimizer, scheduler,
#                 epochs=800, device=device, weights=weights)

#     torch.save(model.state_dict(), "models/best_moe_gru_2 expert.pt")
#     print("模型已保存至 models/best_moe_gru_6 expert.pt")

# if __name__ == '__main__':
#     os.makedirs("models", exist_ok=True)
#     train_once()

# mlp

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
    y = data.iloc[:, :12].values
    X = data.iloc[:, 12:].values

    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)

    y[:, 0] = y[:, 0].astype(int)
    num_samples = X.shape[0]
    X = X.reshape(num_samples, 20)  # ❗️直接使用20维向量作为MLP输入
    return X, y, scaler_x, None

class MLPNetwork(nn.Module):
    def __init__(self, input_dim=20, hidden_dims=[256, 512, 1024, 2048],
                 num_classes=3, num_regression_targets=11):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        self.regressor = nn.Linear(hidden_dims[-1], num_regression_targets)

    def forward(self, x):  # x shape: (batch_size, 20)
        features = self.mlp(x)
        class_out = self.classifier(features)
        raw_regress_out = self.regressor(features)

        regress_out = torch.zeros_like(raw_regress_out)
        regress_out[:, 0] = torch.sigmoid(raw_regress_out[:, 0]) * 8
        regress_out[:, 1:4] = torch.sigmoid(raw_regress_out[:, 1:4]) * 0.2 + 0.05
        regress_out[:, 4:7] = torch.sigmoid(raw_regress_out[:, 4:7]) * 0.5
        regress_out[:, 7] = torch.sigmoid(raw_regress_out[:, 7]) * 10 + 10
        regress_out[:, 8:11] = torch.sigmoid(raw_regress_out[:, 8:11]) * 2 + 1

        return class_out, regress_out, None

def compute_weighted_mse_loss(pred, target, weights):
    squared_error = (pred - target) ** 2
    weighted_squared_error = squared_error * weights.unsqueeze(0)
    return torch.mean(weighted_squared_error)

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=10, device="cuda", weights=None):
    criterion_class = nn.CrossEntropyLoss()
    model.to(device)
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)  # (batch_size, 20)
            y_batch = y_batch.to(device)
            y_class = y_batch[:, 0].long()
            y_regress = y_batch[:, 1:]

            optimizer.zero_grad()
            out_class, out_regress, _ = model(X_batch)

            loss_class = criterion_class(out_class, y_class)
            loss_regress = compute_weighted_mse_loss(out_regress, y_regress, weights)
            loss = loss_class + loss_regress
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        avg_train_loss = epoch_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                y_class = y_val[:, 0].long()
                y_regress = y_val[:, 1:]

                out_class, out_regress, _ = model(X_val)

                loss_class = criterion_class(out_class, y_class)
                loss_regress = compute_weighted_mse_loss(out_regress, y_regress, weights)
                loss = loss_class + loss_regress

                val_loss += loss.item() * X_val.size(0)
                total_samples += X_val.size(0)
                val_mae += torch.sum(torch.abs(out_regress - y_regress)).item()

        val_loss /= total_samples
        val_mae /= total_samples
        scheduler.step(val_loss)

        end_time = time.time()
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"- Train Loss: {avg_train_loss:.6f} "
              f"- Val Loss: {val_loss:.6f} "
              f"- Val MAE: {val_mae:.6f} "
              f"- LR: {optimizer.param_groups[0]['lr']:.6f} "
              f"- Time: {end_time - start_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    return best_val_loss

def train_once():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, scaler_x, _ = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader([
        (torch.tensor(X_train[i], dtype=torch.float32),
         torch.tensor(y_train[i], dtype=torch.float32)) for i in range(len(X_train))
    ], batch_size=512, shuffle=True)

    val_loader = DataLoader([
        (torch.tensor(X_val[i], dtype=torch.float32),
         torch.tensor(y_val[i], dtype=torch.float32)) for i in range(len(X_val))
    ], batch_size=512, shuffle=False)

    regression_std = np.std(y_train[:, 1:], axis=0)
    weights = torch.tensor([1 / std for std in regression_std], device=device)

    model = MLPNetwork(
        input_dim=20,
        hidden_dims=[256, 512, 1024, 2048],
        num_classes=3,
        num_regression_targets=11
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=25, min_lr=1e-6
    )

    train_model(model, train_loader, val_loader, optimizer, scheduler,
                epochs=800, device=device, weights=weights)

    torch.save(model.state_dict(), "models/best_mlp.pt")
    print("模型已保存至 models/best_mlp.pt")

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    train_once()


# 没有MOE
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import os
# import time
# import numpy as np
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # ========================
# # 数据加载 & 预处理
# # ========================
# def load_data():
#     data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
#     y = data.iloc[:, :12].values
#     X = data.iloc[:, 12:].values

#     # 输入标准化
#     scaler_x = StandardScaler()
#     X = scaler_x.fit_transform(X)

#     y[:, 0] = y[:, 0].astype(int)  # 分类标签，其他回归部分不再标准化
#     num_samples = X.shape[0]
#     X = X.reshape(num_samples, 20, 1)
#     return X, y, scaler_x, None  # 不返回 scaler_y

# # ========================
# # GRU Expert
# # ========================
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

#         # 映射到指定范围
#         regress_out = torch.zeros_like(raw_regress_out)
#         regress_out[:, 0] = torch.sigmoid(raw_regress_out[:, 0]) * 8
#         regress_out[:, 1:4] = torch.sigmoid(raw_regress_out[:, 1:4]) * 0.2 + 0.05
#         regress_out[:, 4:7] = torch.sigmoid(raw_regress_out[:, 4:7]) * 0.5
#         regress_out[:, 7] = torch.sigmoid(raw_regress_out[:, 7]) * 10 + 10
#         regress_out[:, 8:11] = torch.sigmoid(raw_regress_out[:, 8:11]) * 2 + 1

#         return class_out, regress_out

# # ========================
# # 加权 MSE 损失
# # ========================
# def compute_weighted_mse_loss(pred, target, weights):
#     """
#     加权 MSE 损失
#     :param pred: 预测值 (batch_size, num_regression_targets)
#     :param target: 真实值 (batch_size, num_regression_targets)
#     :param weights: 权重 (num_regression_targets,)
#     :return: 加权 MSE 损失
#     """
#     squared_error = (pred - target) ** 2
#     weighted_squared_error = squared_error * weights.unsqueeze(0)
#     return torch.mean(weighted_squared_error)

# # ========================
# # 训练函数
# # ========================
# def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=10, device="cuda", weights=None):
#     criterion_class = nn.CrossEntropyLoss()

#     model.to(device)
#     best_val_loss = float('inf')
#     best_model_state = None

#     for epoch in range(epochs):
#         start_time = time.time()
#         model.train()
#         epoch_loss = 0.0

#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)
#             y_class = y_batch[:, 0].long()
#             y_regress = y_batch[:, 1:]

#             optimizer.zero_grad()
#             out_class, out_regress = model(X_batch)
#             loss_class = criterion_class(out_class, y_class)
#             loss_regress = compute_weighted_mse_loss(out_regress, y_regress, weights)
#             loss = loss_class + loss_regress
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item() * X_batch.size(0)

#         avg_train_loss = epoch_loss / len(train_loader.dataset)

#         # 验证阶段
#         model.eval()
#         val_loss = 0.0
#         val_mae = 0.0
#         total_samples = 0

#         with torch.no_grad():
#             for X_val, y_val in val_loader:
#                 X_val = X_val.to(device)
#                 y_val = y_val.to(device)
#                 y_class = y_val[:, 0].long()
#                 y_regress = y_val[:, 1:]

#                 out_class, out_regress = model(X_val)
#                 loss_class = criterion_class(out_class, y_class)
#                 loss_regress = compute_weighted_mse_loss(out_regress, y_regress, weights)
#                 loss = loss_class + loss_regress

#                 val_loss += loss.item() * X_val.size(0)
#                 total_samples += X_val.size(0)

#                 # 直接计算 MAE
#                 val_mae += torch.sum(torch.abs(out_regress - y_regress)).item()

#         val_loss /= total_samples
#         val_mae /= total_samples
#         scheduler.step(val_loss)

#         end_time = time.time()
#         epoch_duration = end_time - start_time

#         print(f"Epoch [{epoch+1}/{epochs}] "
#               f"- Train Loss: {avg_train_loss:.6f} "
#               f"- Val Loss: {val_loss:.6f} "
#               f"- Val MAE: {val_mae:.6f} "
#               f"- LR: {optimizer.param_groups[0]['lr']:.6f} "
#               f"- Time: {epoch_duration:.2f}s")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_state = model.state_dict().copy()

#     model.load_state_dict(best_model_state)
#     return best_val_loss

# # ========================
# # 训练入口
# # ========================
# def train_once():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     X, y, scaler_x, _ = load_data()
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     train_loader = DataLoader([
#         (torch.tensor(X_train[i], dtype=torch.float32),
#          torch.tensor(y_train[i], dtype=torch.float32)) for i in range(len(X_train))
#     ], batch_size=512, shuffle=True)

#     val_loader = DataLoader([
#         (torch.tensor(X_val[i], dtype=torch.float32),
#          torch.tensor(y_val[i], dtype=torch.float32)) for i in range(len(X_val))
#     ], batch_size=512, shuffle=False)

#     # 计算回归目标的权重（基于标准差）
#     regression_std = np.std(y_train[:, 1:], axis=0)
#     weights = torch.tensor([1 / std for std in regression_std], device=device)

#     # 使用单一 GRU 模型
#     model = GRUExpert(
#         input_dim=1,
#         hidden_dims=[256, 512, 1024, 2048],
#         num_classes=3,
#         num_regression_targets=11
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=25, min_lr=1e-6
#     )

#     train_model(model, train_loader, val_loader, optimizer, scheduler,
#                 epochs=800, device=device, weights=weights)

#     torch.save(model.state_dict(), "models/best_gru_without MOE.pt")
#     print("模型已保存至 models/best_gru_multitask_model.pt")

# # ========================
# # 执行主训练
# # ========================
# if __name__ == '__main__':
#     os.makedirs("models", exist_ok=True)
#     train_once()

