# # overal_best
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2Config
from deberta_gru import DeBERTa_MoE_Model, TextDataset  # 替换为你实际保存模型定义的模块名
from gru_MoE_best import GRUMoENetwork  # 替换为你实际保存模型定义的模块名
from tqdm import tqdm
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# ========== 1. 载入 Scaler ==========
def load_scalers():
    # DeBERTa 模型的 scaler_y
    with open('data-augmented2.txt', 'r') as f:
        labels = [list(map(float, line.strip().split(',')))[12:] for line in f if line.strip()]
    scaler_y_deberta = StandardScaler()
    scaler_y_deberta.fit(np.array(labels))

    # GRU-MoE 模型的 scaler_x
    data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
    X_raw = data.iloc[:, 12:].values  # stress-strain curves (输入特征)
    scaler_x_gru = StandardScaler()
    scaler_x_gru.fit(X_raw)

    return labels, scaler_y_deberta, scaler_x_gru

# ========== 2. 载入模型 ==========
def load_models(device="cpu"):
    # 加载 DeBERTa 模型
    config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-base")
    deberta_moe_model = DeBERTa_MoE_Model(config, output_dim=20).to(device)
    checkpoint_path = "finetuned_deberta_gru_8 expert/model.safetensors"
    state_dict = load_file(checkpoint_path)
    deberta_moe_model.load_state_dict(state_dict, strict=True)
    deberta_moe_model.eval()

    # 加载 GRU-MoE 模型
    gru_moe_model = GRUMoENetwork(
        input_dim=1,
        hidden_dims=[256, 512, 1024, 2048],  # 必须与训练时一致
        num_classes=3,
        num_regression_targets=11,
        num_experts=8
    )
    gru_moe_model.load_state_dict(torch.load("models/best_moe_gru_8 expert.pt", map_location=device))
    gru_moe_model.to(device).eval()

    return deberta_moe_model, gru_moe_model

# ========== 3. 执行 DeBERTa 推理 ==========
def run_deberta_inference(batch, deberta_moe_model, scaler_y, device="cpu"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    number_mask = batch['number_mask'].to(device)
    with torch.no_grad():
        outputs = deberta_moe_model(input_ids, attention_mask, number_mask)
        y_target_scaled = outputs["logits"]
    y_target = scaler_y.inverse_transform(y_target_scaled.cpu().numpy())
    return y_target

# ========== 4. 执行 GRU-MoE 推理 ==========
def run_gru_inference(x_features, gru_moe_model, scaler_x, device="cpu"):
    # 标准化输入特征
    x_inputs_scaled = scaler_x.transform(x_features)  # shape: [batch_size, 20]
    x_inputs_tensor = torch.tensor(x_inputs_scaled, dtype=torch.float32).to(device)

    # 将输入数据 reshape 为 [batch_size, sequence_length, input_dim]
    x_inputs_tensor = x_inputs_tensor.unsqueeze(-1)  # shape: [batch_size, 20, 1]

    with torch.no_grad():
        class_output, regress_output, gate_weights = gru_moe_model(x_inputs_tensor)

    # 分类结果
    class_labels = torch.argmax(class_output, dim=1).cpu().numpy()

    # 回归结果：已映射到物理范围，不需要反标准化
    regress_output = regress_output.cpu().numpy()

    return class_labels, regress_output

# ========== 5. 主函数 ==========
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 载入 Scaler 和模型
    labels, scaler_y_deberta, scaler_x_gru = load_scalers()
    deberta_moe_model, gru_moe_model = load_models(device)

    # === 载入文本数据
    data = pd.read_csv('text_qwen.txt', delimiter="\t", header=None)
    texts = data.iloc[:1160, 0].values

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    dataset = TextDataset(texts, labels, tokenizer=tokenizer, max_length=512)

    batch_size = 256
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # === 第一步：使用 DeBERTa 模型生成应力-应变曲线
    y_reconstructed_all = []
    for batch in tqdm(dataloader, desc="DeBERTa 推理中"):
        y_reconstructed = run_deberta_inference(batch, deberta_moe_model, scaler_y_deberta, device)
        y_reconstructed_all.append(y_reconstructed)

    y_reconstructed_all = np.vstack(y_reconstructed_all)

    # === 第二步：使用 GRU-MoE 模型对生成的曲线进行逆向设计
    predicted_class_labels = []
    predicted_regress_outputs = []

    for i in tqdm(range(0, len(y_reconstructed_all), batch_size), desc="GRU-MoE 推理中"):
        x_batch = y_reconstructed_all[i:i + batch_size]
        class_labels, regress_output = run_gru_inference(x_batch, gru_moe_model, scaler_x_gru, device)
        predicted_class_labels.extend(class_labels)
        predicted_regress_outputs.extend(regress_output)

    predicted_class_labels = np.array(predicted_class_labels)
    predicted_regress_outputs = np.array(predicted_regress_outputs)

    # === 计算指标
    true_data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
    true_class_labels = true_data.iloc[:, 0].astype(int).values[:len(predicted_class_labels)]
    true_regress_targets = true_data.iloc[:, 1:12].values[:len(predicted_regress_outputs)]

    classification_accuracy = accuracy_score(true_class_labels, predicted_class_labels)
    regression_rmse = np.sqrt(mean_squared_error(true_regress_targets, predicted_regress_outputs))
    regression_mae = mean_absolute_error(true_regress_targets, predicted_regress_outputs)

    # 计算 NRMSE
    range_true = np.max(true_regress_targets) - np.min(true_regress_targets)  # 真实值的范围
    nrmse = regression_rmse / range_true  # 归一化均方根误差

    print(f"\n最终推理结果：")
    print(f"分类任务准确度: {classification_accuracy:.4f}")
    print(f"回归任务 RMSE: {regression_rmse:.6f}")
    print(f"回归任务 NRMSE: {nrmse:.6f}")
    print(f"回归任务 MAE: {regression_mae:.6f}")

#     # === 保存预测结果
#     predictions_df = pd.DataFrame(np.hstack([predicted_class_labels.reshape(-1, 1), predicted_regress_outputs]))
#     predictions_df.to_csv("final_predictions_8 expert.csv", index=False, header=False)

# without moe

# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2Config
# from deberta_gru import DeBERTa_MoE_Model, TextDataset  # 替换为你实际保存模型定义的模块名
# from gru_MoE_best import GRUExpert  # 替换为你实际保存模型定义的模块名
# from tqdm import tqdm
# from safetensors.torch import load_file
# from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# # ========== 1. 载入 Scaler ==========
# def load_scalers():
#     # DeBERTa 模型的 scaler_y
#     with open('data-augmented2.txt', 'r') as f:
#         labels = [list(map(float, line.strip().split(',')))[12:] for line in f if line.strip()]
#     scaler_y_deberta = StandardScaler()
#     scaler_y_deberta.fit(np.array(labels))

#     # GRU-MoE 模型的 scaler_x
#     data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
#     X_raw = data.iloc[:, 12:].values  # stress-strain curves (输入特征)
#     scaler_x_gru = StandardScaler()
#     scaler_x_gru.fit(X_raw)

#     return labels, scaler_y_deberta, scaler_x_gru

# # ========== 2. 载入模型 ==========
# def load_models(device="cpu"):
#     # 加载 DeBERTa 模型（无 MoE）
#     config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-base")
#     deberta_model = DeBERTa_MoE_Model(config, output_dim=20).to(device)
#     checkpoint_path = "finetuned_deberta_gru_best_without moe/model.safetensors"
#     state_dict = load_file(checkpoint_path)
#     deberta_model.load_state_dict(state_dict, strict=True)
#     deberta_model.eval()

#     # 加载 GRU 模型（单一专家）
#     gru_model = GRUExpert(
#         input_dim=1,
#         hidden_dims=[256, 512, 1024, 2048],
#         num_classes=3,
#         num_regression_targets=11
#     )
#     gru_model.load_state_dict(torch.load("models/best_gru_without MOE.pt", map_location=device))
#     gru_model.to(device).eval()

#     return deberta_model, gru_model

# # ========== 3. 执行 DeBERTa 推理 ==========
# def run_deberta_inference(batch, deberta_model, scaler_y, device="cpu"):
#     input_ids = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     number_mask = batch['number_mask'].to(device)
#     with torch.no_grad():
#         outputs = deberta_model(input_ids, attention_mask, number_mask)
#         y_target_scaled = outputs["logits"]
#     y_target = scaler_y.inverse_transform(y_target_scaled.cpu().numpy())
#     return y_target

# # ========== 4. 执行 GRU 推理 ==========
# def run_gru_inference(x_features, gru_model, scaler_x, device="cpu"):
#     # 标准化输入特征
#     x_inputs_scaled = scaler_x.transform(x_features)  # shape: [batch_size, 20]
#     x_inputs_tensor = torch.tensor(x_inputs_scaled, dtype=torch.float32).to(device)

#     # 将输入数据 reshape 为 [batch_size, sequence_length, input_dim]
#     x_inputs_tensor = x_inputs_tensor.unsqueeze(-1)  # shape: [batch_size, 20, 1]

#     with torch.no_grad():
#         class_output, regress_output = gru_model(x_inputs_tensor)

#     # 分类结果
#     class_labels = torch.argmax(class_output, dim=1).cpu().numpy()

#     # 回归结果：已映射到物理范围，不需要反标准化
#     regress_output = regress_output.cpu().numpy()

#     return class_labels, regress_output

# # ========== 5. 主函数 ==========
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # === 载入 Scaler 和模型
#     labels, scaler_y_deberta, scaler_x_gru = load_scalers()
#     deberta_model, gru_model = load_models(device)

#     # === 载入文本数据
#     data = pd.read_csv('text_qwen.txt', delimiter="\t", header=None)
#     texts = data.iloc[:116000, 0].values

#     tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
#     dataset = TextDataset(texts, labels, tokenizer=tokenizer, max_length=512)

#     batch_size = 256
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     # === 第一步：使用 DeBERTa 模型生成应力-应变曲线
#     y_reconstructed_all = []
#     for batch in tqdm(dataloader, desc="DeBERTa 推理中"):
#         y_reconstructed = run_deberta_inference(batch, deberta_model, scaler_y_deberta, device)
#         y_reconstructed_all.append(y_reconstructed)

#     y_reconstructed_all = np.vstack(y_reconstructed_all)

#     # === 第二步：使用 GRU 模型对生成的曲线进行逆向设计
#     predicted_class_labels = []
#     predicted_regress_outputs = []

#     for i in tqdm(range(0, len(y_reconstructed_all), batch_size), desc="GRU 推理中"):
#         x_batch = y_reconstructed_all[i:i + batch_size]
#         class_labels, regress_output = run_gru_inference(x_batch, gru_model, scaler_x_gru, device)
#         predicted_class_labels.extend(class_labels)
#         predicted_regress_outputs.extend(regress_output)

#     predicted_class_labels = np.array(predicted_class_labels)
#     predicted_regress_outputs = np.array(predicted_regress_outputs)

#     # === 计算指标（如果需要）
#     true_data = pd.read_csv('data-augmented2.txt', delimiter=",", header=None)
#     true_class_labels = true_data.iloc[:, 0].astype(int).values[:len(predicted_class_labels)]
#     true_regress_targets = true_data.iloc[:, 1:12].values[:len(predicted_regress_outputs)]

#     classification_accuracy = accuracy_score(true_class_labels, predicted_class_labels)
#     regression_rmse = np.sqrt(mean_squared_error(true_regress_targets, predicted_regress_outputs))
#     regression_mae = mean_absolute_error(true_regress_targets, predicted_regress_outputs)
#     range_true = np.max(true_regress_targets) - np.min(true_regress_targets)
#     nrmse = regression_rmse / range_true

#     print(f"\n最终推理结果：")
#     print(f"分类任务准确度: {classification_accuracy:.4f}")
#     print(f"回归任务 RMSE: {regression_rmse:.6f}")
#     print(f"回归任务 NRMSE: {nrmse:.6f}")
#     print(f"回归任务 MAE: {regression_mae:.6f}")

#     # === 保存预测结果
#     predictions_df = pd.DataFrame(np.hstack([predicted_class_labels.reshape(-1, 1), predicted_regress_outputs]))
#     predictions_df.to_csv("final_predictions_without moe.csv", index=False, header=False)

