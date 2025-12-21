import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# ==============================================================================
# 1. 核心数据生成器 (回归到各种数字 Setting)
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
    target_idx = 1 # Width
    if random.random() > 0.5:
        return {"type": "4. Dense Retrieval", "premise": premise, "hypothesis": f"Width is {vals[1]}.", "expected": "Entailment"}
    else:
        return {"type": "4. Dense Retrieval", "premise": premise, "hypothesis": f"Width is {vals[1]+1}.", "expected": "Contradiction"}

def get_inequality_case():
    score = random.randint(85, 99)
    threshold = score - 5
    return {"type": "5. Inequality (>)", "premise": f"Score: {score}.", "hypothesis": f"Score more than {threshold}.", "expected": "Entailment"}

def get_chinese_numeral_case():
    map_cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
    val = random.randint(1, 9)
    return {"type": "6. Cross-Lingual (CN)", "premise": f"Found {val} items.", "hypothesis": f"Found {map_cn[val]} items.", "expected": "Entailment"}

# ==============================================================================
# 2. 动态 ID 组装引擎 (Inverse Injection: Fact at Start)
# ==============================================================================

def build_inputs_at_start(tokenizer, case, max_len=512, device="cpu"):
    """
    结构: [CLS] + [Core Premise] + [Filler (Max)] + [SEP] + [Hypothesis] + [SEP]
    核心事实被放在了最前面，距离 Hypothesis 最远。
    """
    hyp_ids = tokenizer.encode(case['hypothesis'], add_special_tokens=False)
    core_ids = tokenizer.encode(case['premise'], add_special_tokens=False)
    
    # 生成 Filler Pool
    single_filler = "Deep learning models often struggle with long-range dependencies due to the attenuation of attention scores over distance. "
    base_filler_ids = tokenizer.encode(single_filler, add_special_tokens=False)
    filler_ids_pool = base_filler_ids * 30 # 足够长
    
    # 计算预算
    # Reserved: CLS(1) + SEP(1) + SEP(1) = 3
    reserved = 3 + len(hyp_ids) + len(core_ids)
    filler_budget = max_len - reserved
    if filler_budget < 0: filler_budget = 0
    
    # 截取 Filler
    current_filler_ids = filler_ids_pool[:filler_budget]
    
    # === 关键组装顺序 ===
    # [CLS] + [CORE] + [FILLER] + [SEP] + [HYP] + [SEP]
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    
    input_ids_list = [cls_id] + core_ids + current_filler_ids + [sep_id] + hyp_ids + [sep_id]
    
    # 截断保护
    if len(input_ids_list) > max_len:
        input_ids_list = input_ids_list[:max_len]
        input_ids_list[-1] = sep_id
        
    input_ids_tensor = torch.tensor([input_ids_list]).to(device)
    attention_mask = torch.ones_like(input_ids_tensor).to(device)
    
    return {"input_ids": input_ids_tensor, "attention_mask": attention_mask}

# ==============================================================================
# 3. 测试运行
# ==============================================================================

def run_start_pos_stress_test(samples=20):
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
    print(f"EXTREME TEST: Core Fact is at the START (Distance ~500 tokens)")
    print(f"{'Type':<25} | {'Tokens':<6} | {'Core Premise (At Start)':<30} | {'Hypothesis':<25} | {'Exp':<5} | {'Pred':<5} | {'Res'}")
    print("="*120)

    stats = {}

    for gen_func in generators:
        temp = gen_func()
        t_name = temp['type']
        stats[t_name] = {"correct": 0, "total": 0}
        
        for _ in range(samples):
            case = gen_func()
            
            # 使用“开头注入”模式
            inputs = build_inputs_at_start(tokenizer, case, max_len=512, device=device)
            actual_len = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = model.config.id2label[pred_id]
            is_correct = (pred_label.lower() == case['expected'].lower())
            
            stats[t_name]['total'] += 1
            if is_correct: stats[t_name]['correct'] += 1
            
            if stats[t_name]['total'] <= 2:
                icon = "✅" if is_correct else "❌"
                print(f"{t_name:<25} | {actual_len:<6} | {case['premise'][:30]:<30} | {case['hypothesis'][:25]:<25} | {case['expected'][:4]:<5} | {pred_label[:4]:<5} | {icon}")

    print("\n" + "="*80)
    print("📊 Final Report: Target at START vs END (Comparison)")
    print("="*80)
    
    # 这里的 Benchmarks 是基于之前实验 "Clumped (All at End)" 的近似值
    benchmarks = {
        "1. Interval Logic": "55%", # 之前本来就低，看看会不会更低
        "2. High Precision Decimals": "85%",
        "3. Large Number Formats": "75%",
        "4. Dense Retrieval": "100%", # 这个之前是满分，看这次能不能破防
        "5. Inequality (>)": "100%",
        "6. Cross-Lingual (CN)": "0%"
    }
    
    print(f"{'Type':<30} | {'At END (Prev)':<15} | {'At START (Current)':<20}")
    print("-" * 70)
    
    for t_name, data in stats.items():
        acc = (data['correct'] / data['total']) * 100
        prev_bench = benchmarks.get(t_name, "N/A")
        print(f"{t_name:<30} | {prev_bench:<15} | {acc:.1f}%")
        
    print("="*80)

if __name__ == "__main__":
    run_start_pos_stress_test(samples=30)