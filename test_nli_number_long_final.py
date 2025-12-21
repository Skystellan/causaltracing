import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# ==============================================================================
# 1. 核心数据生成器 (保持不变)
# ==============================================================================

def get_decimal_case():
    # 生成 5 位小数
    base = round(random.uniform(1, 10), 5)
    # 确保字符串格式统一，防止浮点数精度问题
    base_str = f"{base:.5f}"
    
    # 稍微改动最后一位
    distorted = base + 0.00001
    distorted_str = f"{distorted:.5f}"
    
    return {
        "type": "2. High Precision Decimals",
        "premise": f"The scientific measurement was recorded precisely at {base_str} units.",
        "hypothesis": f"The scientific measurement was recorded precisely at {distorted_str} units.",
        "expected": "Contradiction",
        "key_snippet": base_str # 用于验证完整性的关键片段
    }

# ... 其他生成器保持一致，为了节省篇幅，这里只列出这一种最容易出错的作为代表 ...
# 在实际运行中，你可以把之前所有的生成器都加回来

def get_interval_case():
    age = random.randint(20, 59)
    decade = (age // 10) * 10
    wrong_decade = decade + 10
    if random.random() > 0.5:
        return {"type": "1. Interval Logic", "premise": f"The suspect is currently {age} years old.", "hypothesis": f"The suspect is in his {decade}s.", "expected": "Entailment", "key_snippet": str(age)}
    else:
        return {"type": "1. Interval Logic", "premise": f"The suspect is currently {age} years old.", "hypothesis": f"The suspect is in his {wrong_decade}s.", "expected": "Contradiction", "key_snippet": str(age)}

def get_large_format_case():
    val = random.randint(1, 9)
    half = random.choice([0, 5])
    num_str = f"{val},{half}00,000"
    word_str = f"{val}.{half} million"
    wrong_word_str = f"{val + 1}.{half} million"
    if random.random() > 0.5:
        return {"type": "3. Large Number Formats", "premise": f"Cost: {num_str} dollars.", "hypothesis": f"Cost: {word_str} dollars.", "expected": "Entailment", "key_snippet": num_str}
    else:
        return {"type": "3. Large Number Formats", "premise": f"Cost: {num_str} dollars.", "hypothesis": f"Cost: {wrong_word_str} dollars.", "expected": "Contradiction", "key_snippet": num_str}

def get_dense_info_case():
    metrics = ["Length", "Width", "Height"]
    vals = [random.randint(10, 50) for _ in range(3)]
    premise = f"Length is {vals[0]}, Width is {vals[1]}, Height is {vals[2]}."
    target_idx = 1
    if random.random() > 0.5:
        return {"type": "4. Dense Retrieval", "premise": premise, "hypothesis": f"Width is {vals[1]}.", "expected": "Entailment", "key_snippet": str(vals[2])} # 验证最后一个数字在不在
    else:
        return {"type": "4. Dense Retrieval", "premise": premise, "hypothesis": f"Width is {vals[1]+1}.", "expected": "Contradiction", "key_snippet": str(vals[2])}

def get_inequality_case():
    score = random.randint(85, 99)
    threshold = score - 5
    return {"type": "5. Inequality (>)", "premise": f"Score: {score}.", "hypothesis": f"Score more than {threshold}.", "expected": "Entailment", "key_snippet": str(score)}

def get_chinese_numeral_case():
    map_cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
    val = random.randint(1, 9)
    return {"type": "6. Cross-Lingual (CN)", "premise": f"Found {val} items.", "hypothesis": f"Found {map_cn[val]} items.", "expected": "Entailment", "key_snippet": str(val)}


# ==============================================================================
# 2. 偏执级安全注入引擎 (Paranoid Safe Injection)
# ==============================================================================

def build_inputs_paranoid(tokenizer, case, max_len=512, device="cpu"):
    """
    1. 目标长度设为 max_len - 5 (安全气囊)
    2. 优先保证 Core 完整
    3. 组装后反向 Decode 验证
    """
    SAFETY_MARGIN = 10 # 留10个token的空余，绝不填满
    target_len = max_len - SAFETY_MARGIN
    
    hyp_ids = tokenizer.encode(case['hypothesis'], add_special_tokens=False)
    core_ids = tokenizer.encode(case['premise'], add_special_tokens=False)
    
    # Filler Pool
    single_filler = "Attention mechanisms allow models to focus on specific parts of the input sequence. "
    base_filler_ids = tokenizer.encode(single_filler, add_special_tokens=False)
    filler_ids_pool = base_filler_ids * 50 
    
    # 计算预算
    # [CLS] + Filler + [Core] + [SEP] + [Hyp] + [SEP]
    reserved = 3 + len(hyp_ids) + len(core_ids)
    filler_budget = target_len - reserved
    
    if filler_budget < 0: filler_budget = 0
    
    # 截取 Filler
    current_filler_ids = filler_ids_pool[:filler_budget]
    
    # 组装
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    
    # 结构: CLS + Filler + CORE + SEP + HYP + SEP
    input_ids_list = [cls_id] + current_filler_ids + core_ids + [sep_id] + hyp_ids + [sep_id]
    
    # === 偏执级检查 ===
    if len(input_ids_list) > max_len:
        print(f"⚠️ 警告：长度溢出 {len(input_ids_list)} > {max_len}。正在紧急削减 Filler...")
        diff = len(input_ids_list) - max_len
        # 削减 Filler
        current_filler_ids = current_filler_ids[:-diff]
        # 重新组装
        input_ids_list = [cls_id] + current_filler_ids + core_ids + [sep_id] + hyp_ids + [sep_id]
        
    input_ids_tensor = torch.tensor([input_ids_list]).to(device)
    attention_mask = torch.ones_like(input_ids_tensor).to(device)
    
    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask,
        "final_list": input_ids_list # 用于Debug
    }

# ==============================================================================
# 3. 运行测试
# ==============================================================================

def run_paranoid_test(samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    generators = [
        get_interval_case,
        get_decimal_case,
        get_large_format_case,
        get_dense_info_case,
        get_inequality_case,
        get_chinese_numeral_case
    ]
    
    print("\n" + "="*120)
    print(f"PARANOID TEST: End Position with INTEGRITY CHECK (Safety Margin=10)")
    print(f"{'Type':<30} | {'Len':<5} | {'Integrity':<10} | {'Exp':<5} | {'Pred':<5} | {'Res'}")
    print("="*120)

    stats = {}

    for gen_func in generators:
        temp = gen_func()
        t_name = temp['type']
        stats[t_name] = {"correct": 0, "total": 0}
        
        for _ in range(samples):
            case = gen_func()
            
            # 构建输入
            inputs_dict = build_inputs_paranoid(tokenizer, case, max_len=512, device=device)
            
            # === 完整性验证 (DEBUG 核心) ===
            input_ids = inputs_dict['final_list']
            decoded_text = tokenizer.decode(input_ids)
            
            # 检查关键片段是否完整存在
            key_snippet = case['key_snippet']
            is_intact = key_snippet in decoded_text
            
            integrity_msg = "OK" if is_intact else "BROKEN!!"
            
            if not is_intact:
                print(f"\n🛑 严重错误！关键信息丢失！")
                print(f"Expected: {key_snippet}")
                print(f"Actual End of Premise: ...{decoded_text[-100:]}")
                # 跳过这次错误的推理，或者记录为失败
            
            # 准备模型输入
            model_inputs = {
                "input_ids": inputs_dict['input_ids'],
                "attention_mask": inputs_dict['attention_mask']
            }
            
            with torch.no_grad():
                outputs = model(**model_inputs)
            
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = model.config.id2label[pred_id]
            is_correct = (pred_label.lower() == case['expected'].lower())
            
            stats[t_name]['total'] += 1
            if is_correct: stats[t_name]['correct'] += 1
            
            if stats[t_name]['total'] <= 2:
                icon = "✅" if is_correct else "❌"
                print(f"{t_name:<30} | {len(input_ids):<5} | {integrity_msg:<10} | {case['expected'][:4]:<5} | {pred_label[:4]:<5} | {icon}")

    print("\n" + "="*80)
    print("📊 Final Accurate Report (No Truncation Guaranteed)")
    print("="*80)
    
    for t_name, data in stats.items():
        acc = (data['correct'] / data['total']) * 100
        print(f"{t_name:<30}: {acc:.1f}% ({data['correct']}/{data['total']})")

if __name__ == "__main__":
    run_paranoid_test(samples=30)