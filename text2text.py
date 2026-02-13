# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     TrainingArguments,
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from trl import SFTTrainer
# # , DataCollatorForCompletionOnlyLM

# # --- 1. 配置参数 ---
# # model_name = "Qwen/Qwen2.5-7B-Instruct"  # 这里换成你实际下载的模型路径
# model_name = "/home/zzs/data/inference/qwen/Qwen2.5-7B-Instruct"
# dataset_file = "dataset_train.json"
# output_dir = "./qwen_finetuned_output"

# # --- 2. 加载数据集 ---
# dataset = load_dataset("json", data_files=dataset_file, split="train")

# # --- 3. 加载模型和 Tokenizer (4bit 量化以节省显存) ---
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token # Qwen 的 pad token 设置

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )

# # 开启梯度检查点，节省显存
# model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model)

# # --- 4. 配置 LoRA ---
# peft_config = LoraConfig(
#     r=16,     # LoRA 秩，越大参数越多，拟合能力越强，显存消耗越大（建议 8-64）
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     # Qwen 的核心模块，全部微调效果最好
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# )

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters() # 打印可训练参数量

# # --- 5. 定义训练参数 ---
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=4, # 显存不够就调小，比如 2 或 1
#     gradient_accumulation_steps=4, # 累积梯度，相当于变相增大 batch size
#     learning_rate=2e-4,            # LoRA 学习率通常比全量微调大
#     logging_steps=10,
#     num_train_epochs=3,            # 训练轮数，数据少可以适当增加到 5-10
#     save_strategy="epoch",
#     fp16=True,                     # 开启混合精度训练
#     optim="paged_adamw_32bit",     # 节省显存的优化器
#     report_to="none"               # 不上传到 wandb
# )

# # --- 6. 训练 Trainer ---
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     args=training_args,
#     # tokenizer=tokenizer,
#     peft_config=peft_config,
#     # max_seq_length=512, # 根据你的文本长度调整，越长越占显存
# )

# # --- 7. 开始训练 ---
# print("开始训练...")
# trainer.train()

# # --- 8. 保存模型 ---
# print(f"训练完成，保存模型至 {output_dir}")
# trainer.model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     TrainingArguments,
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# # ================= 配置区域 =================
# # 模型路径 (请确认路径无误)
# model_name = "/home/zzs/data/inference/qwen/Qwen2.5-7B-Instruct"
# # 数据集路径 (请确保此前已经运行过数据处理脚本生成了该文件)
# dataset_file = "dataset_train.json"
# # 输出路径
# output_dir = "./qwen_finetuned_output1"

# # ================= 1. 加载数据集 =================
# dataset = load_dataset("json", data_files=dataset_file, split="train")

# # ================= 2. 加载模型和 Tokenizer =================
# print("正在加载模型和 Tokenizer...")
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token 

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )

# # 开启梯度检查点以节省显存
# model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model)

# # ================= 3. 配置 LoRA =================
# peft_config = LoraConfig(
#     r=16, 
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# )

# model = get_peft_model(model, peft_config)
# print(">>> 可训练参数量:")
# model.print_trainable_parameters()

# # ================= 4. 定义格式化函数 (修复 ValueError 的关键) =================
# def formatting_prompts_func(example):
#     output_texts = []
#     for messages in example['messages']:
#         # 使用 Qwen 的 chat template 将 messages 列表转为字符串
#         text = tokenizer.apply_chat_template(
#             messages, 
#             tokenize=False, 
#             add_generation_prompt=False
#         )
#         output_texts.append(text)
#     return output_texts

# # ================= 5. 定义 Data Collator (只计算回复部分的 Loss) =================
# # Qwen 的回复前缀通常是 "<|im_start|>assistant\n"
# response_template = "<|im_start|>assistant\n"
# collator = DataCollatorForCompletionOnlyLM(
#     response_template=response_template, 
#     tokenizer=tokenizer
# )

# # ================= 6. 定义训练参数 (针对精确度优化) =================
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=2,  # 显存允许的话可改为 4
#     gradient_accumulation_steps=4,  # 累计梯度
#     learning_rate=5e-5,             # 【关键】降低学习率，防止在精确数值附近震荡
#     logging_steps=10,
#     num_train_epochs=10,            # 【关键】增加轮数，强制模型记住映射逻辑
#     save_strategy="epoch",
#     fp16=True, 
#     optim="paged_adamw_32bit",
#     report_to="none"
# )

# # ================= 7. 初始化 Trainer =================
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     args=training_args,
#     peft_config=peft_config,
#     data_collator=collator,                  # 使用 CompletionOnlyLM
#     formatting_func=formatting_prompts_func, # 【关键】传入格式化函数
#     max_seq_length=1024,
# )

# # ================= 8. 开始训练 =================
# print("开始训练 (Target: High Numerical Precision)...")
# trainer.train()

# # ================= 9. 保存模型 =================
# print(f"训练完成，模型已保存至 {output_dir}")
# trainer.model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# 最终完整的训练模型
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ================= 配置区域 =================
model_name = "/home/zzs/data/inference/qwen/Qwen2.5-7B-Instruct"
dataset_file = "dataset_train.json"
output_dir = "./qwen_finetuned_weighted" # 修改输出路径以示区别

# --- 关键超参数 ---
NUMBER_LOSS_WEIGHT = 5.0  # 【核心】数字错误的惩罚倍数 (建议 3.0 - 10.0)
LEARNING_RATE = 5e-5      # 保持较低的学习率
NUM_EPOCHS = 10          # 保持较多轮次

# ================= 1. 自定义 Trainer (实现加权 Loss) =================
class NumberWeightedSFTTrainer(SFTTrainer):
    def __init__(self, *args, number_weight=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_weight = number_weight
        
        # 预处理：找到词表中所有包含数字的 Token ID
        print(">>> 正在构建数字 Token 索引 (用于加权 Loss)...")
        vocab = self.tokenizer.get_vocab()
        self.number_token_ids = set()
        for token, id in vocab.items():
            # Qwen 的 token 有时是 byte 编码，解码后检查是否含数字
            # 简单粗暴的方法：检查 token 字符串中是否含 '0'-'9'
            # 注意：SentencePiece/BPE token 可能包含前缀，如 " 123"
            try:
                # 尝试解码 token
                decoded = self.tokenizer.decode([id])
                if any(c.isdigit() for c in decoded):
                    self.number_token_ids.add(id)
            except:
                pass
        
        # 将 set 转为 tensor 方便后续计算，移至模型设备会在 compute_loss 中处理
        # 这里先存个 list
        self.number_token_ids_list = list(self.number_token_ids)
        print(f">>> 识别到 {len(self.number_token_ids)} 个包含数字的 Token。")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 Loss 计算逻辑：
        1. 获取原始 logits 和 labels
        2. 计算逐元素的 CrossEntropyLoss (不求平均)
        3. 对属于数字的 token 赋予更高的权重
        4. 求平均并返回
        """
        labels = inputs.get("labels")
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 这里的逻辑参考 CausalLM 的标准 loss 计算
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 1. 计算原始 Loss (reduction='none' 是关键，我们要拿到每个 token 的 loss)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # 展平计算
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # 此时 loss 是一个形状为 [batch_size * seq_len] 的向量
        loss = loss_fct(shift_logits, shift_labels)
        
        # 2. 构建权重掩码 (Weight Mask)
        # 初始化权重为 1.0
        weights = torch.ones_like(loss)
        
        # 找到所有 label 是数字的地方
        #为了加速，我们需要将 self.number_token_ids 转为当前 device 的 tensor
        if not hasattr(self, 'number_token_tensor') or self.number_token_tensor.device != loss.device:
            self.number_token_tensor = torch.tensor(
                self.number_token_ids_list, device=loss.device, dtype=shift_labels.dtype
            )
            
        # 判断 shift_labels 中的元素是否存在于 number_token_tensor 中
        # torch.isin 是最快的方法
        is_number_mask = torch.isin(shift_labels, self.number_token_tensor)
        
        # 3. 应用权重
        # 如果是数字，权重设为 self.number_weight (比如 5.0)，否则保持 1.0
        weights = torch.where(is_number_mask, self.number_weight, 1.0)
        
        # 注意：DataCollator 可能会把 pad 或者 prompt 部分的 label 设为 -100
        # CrossEntropyLoss 默认已经忽略了 -100 的 loss (变为 0)
        # 我们这里不需要额外处理 -100，因为 0 * weight 还是 0
        
        # 4. 加权并求平均
        weighted_loss = loss * weights
        
        # 求平均时，分母应该是有效 token 的数量，或者是有效权重之和
        # 为了保持训练稳定，建议除以有效 token 数量 (非 -100 的数量)
        # 或者直接 mean()，因为 loss 中非有效位已经是 0 了
        final_loss = weighted_loss.sum() / (shift_labels != -100).sum()
        
        return (final_loss, outputs) if return_outputs else final_loss

# ================= 2. 常规设置 (同方案一) =================
dataset = load_dataset("json", data_files=dataset_file, split="train")

print("正在加载模型...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 格式化函数
def formatting_prompts_func(example):
    output_texts = []
    for messages in example['messages']:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)
    return output_texts

# Data Collator (方案一的内容，保留)
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

# ================= 3. 训练参数 =================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,    # 5e-5
    logging_steps=10,
    num_train_epochs=NUM_EPOCHS,    # 10
    save_strategy="epoch",
    fp16=True, 
    optim="paged_adamw_32bit",
    report_to="none"
)

# ================= 4. 初始化自定义 Trainer =================
print(f"初始化 Trainer，数字权重设置为: {NUMBER_LOSS_WEIGHT} 倍")
trainer = NumberWeightedSFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    data_collator=collator,
    formatting_func=formatting_prompts_func,
    max_seq_length=1024,
    number_weight=NUMBER_LOSS_WEIGHT  # 传入权重参数
)

# ================= 5. 开始训练 =================
print("开始训练 (Scheme 4: Weighted Loss for Numbers)...")
trainer.train()

print(f"训练完成，模型已保存至 {output_dir}")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

from modelscope import snapshot_download

# 下载模型到当前目录下的 Qwen2.5-7B-Instruct 文件夹
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='./')
print(f"模型已下载到: {model_dir}")


# 推理部分
import torch
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ================= 配置区域 =================
# 路径设置 (请修改为你自己的路径)
BASE_MODEL_PATH = "/home/zzs/data/inference/qwen/Qwen2.5-7B-Instruct"  # 基座模型
ADAPTER_PATH = "./qwen_finetuned_weighted"     # 微调后的权重
INPUT_FILE = "text_robust_out.txt"           # 模糊文本 (User Input)
TARGET_FILE = "text_robust_in.txt"           # 精确文本 (Ground Truth)

# 测试样本数 (None 表示测试全部，建议先测 50 条看看效果)
TEST_SAMPLES = None

# 生成参数
GEN_CONFIG = {
    "max_new_tokens": 128,  # 不需要太长，只要数值出来就行
    "temperature": 0.1,     # 低温，保证稳定性
    "top_p": 0.9
}
# ===========================================

def load_data(input_path, target_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(target_path, 'r', encoding='utf-8') as f_out:
        inputs = [line.strip() for line in f_in.readlines() if line.strip()]
        targets = [line.strip() for line in f_out.readlines() if line.strip()]
    min_len = min(len(inputs), len(targets))
    return list(zip(inputs[:min_len], targets[:min_len]))

def extract_stiffness(text):
    """
    使用正则表达式提取文本中的 Stiffness 数值
    支持格式: "stiffness of 0.70", "stiffness of 1.61"
    """
    # 匹配 "stiffness of" 后面紧跟的数字 (支持整数和小数)
    pattern = r"stiffness of\s*(\d+\.?\d*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def analyze_prediction(fuzzy_input, gt_text, pred_text):
    """
    智能分析函数：对比真实值和预测值
    """
    gt_val = extract_stiffness(gt_text)
    pred_val = extract_stiffness(pred_text)
    
    result = {
        "status": "Unknown",
        "gt_val": gt_val,
        "pred_val": pred_val,
        "diff": 0.0,
        "msg": ""
    }

    # 1. 如果提取失败
    if gt_val is None or pred_val is None:
        result["status"] = "⚠️ 解析失败"
        result["msg"] = "未能从文本中提取到 stiffness 数值"
        return result

    # 2. 计算差异
    diff = abs(gt_val - pred_val)
    result["diff"] = diff

    # 3. 判定逻辑
    # 判定 A: 精确命中 (误差小于 0.3 或 相对误差小于 15%)
    if diff <= 0.5 or (gt_val > 0 and diff / gt_val < 0.3):
        result["status"] = "✅ 精确命中"
        result["msg"] = f"误差仅 {diff:.2f}"
    
    # 判定 B: 趋势正确 (量级判断)
    # 假设: Soft < 2.0, Moderate 2.0-5.0, Stiff > 5.0 (根据你的数据分布调整)
    elif (gt_val < 2.0 and pred_val < 2.0) or \
         (gt_val > 5.0 and pred_val > 5.0):
        result["status"] = "👌 趋势正确"
        result["msg"] = f"数值不同但处于同一强度区间 (GT:{gt_val} vs Pred:{pred_val})"
    
    # 判定 C: 错误
    else:
        result["status"] = "❌ 偏差较大"
        result["msg"] = f"真实值 {gt_val}，预测值 {pred_val}，差异明显"

    return result

# ================= 主程序 =================
def main():
    print(f"🔄 正在加载模型: {BASE_MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("📚 正在加载测试数据...")
    test_data = load_data(INPUT_FILE, TARGET_FILE)
    if TEST_SAMPLES:
        test_data = test_data[:TEST_SAMPLES]
    
    system_prompt = "You are a specialized mechanical engineer. Your task is to translate qualitative, fuzzy structural descriptions into precise quantitative test parameters."
    
    stats = {"exact": 0, "trend": 0, "fail": 0, "parse_error": 0, "diffs": []}
    
    print("\n🚀 开始智能评估...")
    print("="*60)

    for i, (inp, truth) in enumerate(tqdm(test_data)):
        # 构建 Prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": inp}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 推理
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **GEN_CONFIG
            )
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        prediction = output.split("assistant")[-1].strip()

        # 分析
        res = analyze_prediction(inp, truth, prediction)

        # 统计
        if "✅" in res["status"]: stats["exact"] += 1
        elif "👌" in res["status"]: stats["trend"] += 1
        elif "❌" in res["status"]: stats["fail"] += 1
        else: stats["parse_error"] += 1
        
        if res["gt_val"] is not None and res["pred_val"] is not None:
            stats["diffs"].append(res["diff"])

        # 打印部分结果 (每 5 条打印一次，或者是错误的时候打印)
        if i < 5 or "❌" in res["status"]:
            print(f"\n[Sample {i+1}]")
            print(f"📝 输入特征: {inp[:60]}...") # 截断显示
            print(f"📊 评估结果: {res['status']} | {res['msg']}")
            if res["status"] == "❌ 偏差较大":
                print(f"   -> GT: {truth[:80]}")
                print(f"   -> PR: {prediction[:80]}")

    print("="*60)
    print("\n📈 最终评估报告")
    total = len(test_data)
    valid_count = len(stats["diffs"])
    
    print(f"总样本数: {total}")
    print(f"✅ 精确命中: {stats['exact']} ({stats['exact']/total:.1%})")
    print(f"👌 趋势正确: {stats['trend']} ({stats['trend']/total:.1%})")
    print(f"❌ 偏差较大: {stats['fail']} ({stats['fail']/total:.1%})")
    print(f"⚠️ 解析失败: {stats['parse_error']} (无法提取数字)")
    
    if valid_count > 0:
        mae = np.mean(stats["diffs"])
        print(f"\n🔢 平均绝对误差 (MAE): {mae:.4f}")
        print("💡 结论: MAE 越低越好。如果 MAE < 0.5，说明模型在工程上是可用的。")

if __name__ == "__main__":
    main()