import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# ==============================================================================
# 1. 全量数据生成器 (6种类型)
# ==============================================================================

def get_interval_case():
    age = random.randint(20, 59)
    decade = (age // 10) * 10
    wrong_decade = decade + 10
    if random.random() > 0.5:
        return {"type": "1. Interval Logic", "premise": f"The suspect is currently {age} years old.", "hypothesis": f"The suspect is in his {decade}s.", "expected": "Entailment"}
    else:
        return {"type": "1. Interval Logic", "premise": f"The suspect is currently {age} years old.", "hypothesis": f"The suspect is in his {wrong_decade}s.", "expected": "Contradiction"}

def get_decimal_case():
    base = round(random.uniform(1, 10), 5)
    distorted = base + 0.00001
    return {"type": "2. High Precision Decimals", "premise": f"The reading was exactly {base:.5f}.", "hypothesis": f"The reading was exactly {distorted:.5f}.", "expected": "Contradiction"}

def get_large_format_case():
    val = random.randint(1, 9)
    half = random.choice([0, 5])
    num_str = f"{val},{half}00,000"
    word_str = f"{val}.{half} million"
    wrong_word_str = f"{val + 1}.{half} million"
    if random.random() > 0.5:
        return {"type": "3. Large Number Formats", "premise": f"Cost: {num_str} dollars.", "hypothesis": f"Cost: {word_str} dollars.", "expected": "Entailment"}
    else:
        return {"type": "3. Large Number Formats", "premise": f"Cost: {num_str} dollars.", "hypothesis": f"Cost: {wrong_word_str} dollars.", "expected": "Contradiction"}

def get_dense_info_case():
    metrics = ["Length", "Width", "Height"]
    vals = [random.randint(10, 50) for _ in range(3)]
    premise = f"Length is {vals[0]}, Width is {vals[1]}, Height is {vals[2]}."
    # 随机选一个属性测
    target_idx = random.randint(0, 2)
    target_name = metrics[target_idx]
    target_val = vals[target_idx]
    
    if random.random() > 0.5:
        return {"type": "4. Dense Retrieval", "premise": premise, "hypothesis": f"{target_name} is {target_val}.", "expected": "Entailment"}
    else:
        return {"type": "4. Dense Retrieval", "premise": premise, "hypothesis": f"{target_name} is {target_val+1}.", "expected": "Contradiction"}

def get_inequality_case():
    score = random.randint(85, 99)
    threshold = score - 5
    return {"type": "5. Inequality (>)", "premise": f"Score: {score}.", "hypothesis": f"Score more than {threshold}.", "expected": "Entailment"}

def get_chinese_numeral_case():
    map_cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
    val = random.randint(1, 9)
    return {"type": "6. Cross-Lingual (CN)", "premise": f"Found {val} items.", "hypothesis": f"Found {map_cn[val]} items.", "expected": "Entailment"}

# ==============================================================================
# 2. 动态 ID 组装引擎 (Middle Injection)
# ==============================================================================

def build_inputs_at_middle(tokenizer, case, max_len=512, device="cpu"):
    """
    构造: [CLS] + [Filler_Pre] + [CORE] + [Filler_Post] + [SEP] + [HYP] + [SEP]
    核心事实被深埋在正中间。
    """
    hyp_ids = tokenizer.encode(case['hypothesis'], add_special_tokens=False)
    core_ids = tokenizer.encode(case['premise'], add_special_tokens=False)
    
    # Filler Pool (足够长)
    single_filler = "Midway upon the journey of our life I found myself within a forest dark, for the straightforward pathway had been lost. "
    base_filler_ids = tokenizer.encode(single_filler, add_special_tokens=False)
    filler_ids_pool = base_filler_ids * 40
    
    # 预算计算
    # Reserved: CLS(1) + SEP(1) + SEP(1) = 3
    reserved = 3 + len(hyp_ids) + len(core_ids)
    filler_budget = max_len - reserved
    if filler_budget < 0: filler_budget = 0
    
    # 截取总 Filler
    total_filler_ids = filler_ids_pool[:filler_budget]
    
    # 将 Filler 一分为二
    half_idx = len(total_filler_ids) // 2
    filler_pre = total_filler_ids[:half_idx]
    filler_post = total_filler_ids[half_idx:]
    
    # === 组装 ===
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    
    input_ids_list = [cls_id] + filler_pre + core_ids + filler_post + [sep_id] + hyp_ids + [sep_id]
    
    # 安全截断 (防止计算微小误差)
    if len(input_ids_list) > max_len:
        input_ids_list = input_ids_list[:max_len]
        input_ids_list[-1] = sep_id
        
    input_ids_tensor = torch.tensor([input_ids_list]).to(device)
    attention_mask = torch.ones_like(input_ids_tensor).to(device)
    
    return {"input_ids": input_ids_tensor, "attention_mask": attention_mask}

# ==============================================================================
# 3. 执行全量测试
# ==============================================================================

def run_full_middle_test(samples=30):
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
    print(f"TEST CONFIG: Core Fact BURIED IN MIDDLE (Index ~250)")
    print(f"{'Type':<30} | {'Tokens':<6} | {'Core Premise':<30} | {'Hypothesis':<25} | {'Exp':<5} | {'Pred':<5} | {'Res'}")
    print("="*120)

    stats = {}

    for gen_func in generators:
        temp = gen_func()
        t_name = temp['type']
        stats[t_name] = {"correct": 0, "total": 0}
        
        for _ in range(samples):
            case = gen_func()
            inputs = build_inputs_at_middle(tokenizer, case, max_len=512, device=device)
            actual_len = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = model.config.id2label[pred_id]
            is_correct = (pred_label.lower() == case['expected'].lower())
            
            stats[t_name]['total'] += 1
            if is_correct: stats[t_name]['correct'] += 1
            
            # 打印少量 Log
            if stats[t_name]['total'] <= 1:
                icon = "✅" if is_correct else "❌"
                print(f"{t_name:<30} | {actual_len:<6} | {case['premise'][:30]:<30} | {case['hypothesis'][:25]:<25} | {case['expected'][:4]:<5} | {pred_label[:4]:<5} | {icon}")

    print("\n" + "="*100)
    print("📊 Final Comparative Report: Start vs Middle vs End")
    print("="*100)
    
    # 之前测试的数据 (Hardcoded based on your previous outputs)
    # End: 之前的 Clumped/Long Context 测试结果
    # Start: 刚刚跑出来的 100% 结果
    prev_data = {
        "1. Interval Logic":          {"Start": "100%", "End": "55%"},
        "2. High Precision Decimals": {"Start": "93%",  "End": "85%"},
        "3. Large Number Formats":    {"Start": "100%", "End": "75%"},
        "4. Dense Retrieval":         {"Start": "100%", "End": "100%"},
        "5. Inequality (>)":          {"Start": "100%", "End": "100%"},
        "6. Cross-Lingual (CN)":      {"Start": "100%", "End": "0%"} 
    }
    
    print(f"{'Type':<30} | {'Start (Heuristic)':<18} | {'MIDDLE (Current)':<18} | {'End (Recency)':<15}")
    print("-" * 90)
    
    for t_name, data in stats.items():
        mid_acc = (data['correct'] / data['total']) * 100
        start_score = prev_data.get(t_name, {}).get("Start", "?")
        end_score = prev_data.get(t_name, {}).get("End", "?")
        
        print(f"{t_name:<30} | {start_score:<18} | {mid_acc:.1f}%{'':<13} | {end_score:<15}")
        
    print("="*100)

if __name__ == "__main__":
    run_full_middle_test(samples=30)