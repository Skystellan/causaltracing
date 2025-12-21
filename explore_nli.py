import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

def build_noisy_input_safe(tokenizer, core_text, max_len=512, device="cpu"):
    """
    精确控制：[CLS] [Core] [Noise (Max)] [SEP] [Hyp] [SEP]
    确保填满窗口，不报错。
    """
    hyp_text = core_text # Hypothesis 和 Premise 核心一样
    hyp_ids = tokenizer.encode(hyp_text, add_special_tokens=False)
    core_ids = tokenizer.encode(core_text, add_special_tokens=False)
    
    # 制造大量数字干扰
    noise_unit = "Error code 404. System load 99%. Memory address 0x3F. Temp 75C. "
    base_noise_ids = tokenizer.encode(noise_unit, add_special_tokens=False)
    noise_pool = base_noise_ids * 50 # 足够长
    
    # 倒推预算
    reserved = 3 + len(hyp_ids) + len(core_ids)
    noise_budget = max_len - reserved
    if noise_budget < 0: noise_budget = 0
    
    current_noise = noise_pool[:noise_budget]
    
    # 组装：Core 在前，Noise 在后 (模拟 Start 位置)
    cls = [tokenizer.cls_token_id]
    sep = [tokenizer.sep_token_id]
    
    input_ids_list = cls + core_ids + current_noise + sep + hyp_ids + sep
    
    # 转 Tensor
    input_ids_tensor = torch.tensor([input_ids_list]).to(device)
    return input_ids_tensor

def run_final_check(samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    print(f"Re-verifying Noisy Start with PRECISE length control...")
    correct = 0
    for _ in range(samples):
        target_val = random.randint(1000, 9999)
        txt = f"Subject ID is {target_val}."
        
        inputs = build_noisy_input_safe(tokenizer, txt, max_len=512, device=device)
        
        with torch.no_grad():
            out = model(inputs)
        
        pred = model.config.id2label[torch.argmax(out.logits).item()]
        if pred.lower() == "entailment": correct += 1
        
    print(f"Noisy Start Accuracy: {correct/samples*100:.1f}%")

if __name__ == "__main__":
    run_final_check()