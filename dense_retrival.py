import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# ==============================================================================
# 1. 核心数据生成器 (Dense Retrieval: Length/Width/Height)
# ==============================================================================

def get_dense_case():
    metrics = ["Length", "Width", "Height"]
    vals = [random.randint(10, 50) for _ in range(3)]
    
    # 生成三个独立的事实句子
    facts = [f"{m} is {v}." for m, v in zip(metrics, vals)]
    fact_map = {m: f for m, f in zip(metrics, facts)}
    
    # 随机打乱事实列表的顺序 (用于 Scattered 分配位置)
    shuffled_keys = metrics[:]
    random.shuffle(shuffled_keys)
    ordered_facts = [fact_map[k] for k in shuffled_keys]
    
    # 随机选一个作为目标
    target_metric = random.choice(metrics)
    target_val = vals[metrics.index(target_metric)]
    
    # 找到目标在 ordered_facts 中的位置索引
    target_pos_idx = shuffled_keys.index(target_metric)
    pos_map = {0: "Start", 1: "Middle", 2: "End"}
    target_pos_name = pos_map[target_pos_idx]
    
    # 构造 Hypothesis
    if random.random() > 0.5:
        hyp = f"{target_metric} is {target_val}."
        expected = "Entailment"
    else:
        hyp = f"{target_metric} is {target_val + 1}."
        expected = "Contradiction"
        
    return {
        "ordered_facts": ordered_facts,
        "hypothesis": hyp,
        "expected": expected,
        "target_pos": target_pos_name,
        "target_metric": target_metric
    }

# ==============================================================================
# 2. 动态 ID 组装引擎 (绝对安全版)
# ==============================================================================

def build_inputs_precise(tokenizer, case, mode="clumped", max_len=512, device="cpu"):
    # 1. 独立编码各部分 (不带特殊字符)
    facts = case['ordered_facts']
    hyp_ids = tokenizer.encode(case['hypothesis'], add_special_tokens=False)
    
    f0_ids = tokenizer.encode(facts[0], add_special_tokens=False)
    f1_ids = tokenizer.encode(facts[1], add_special_tokens=False)
    f2_ids = tokenizer.encode(facts[2], add_special_tokens=False)
    
    # 2. 生成 Filler Pool (修复点：先编码短句，再复制 ID 列表)
    filler_raw = "System check nominal. Data stream active. Monitoring parameters. "
    base_filler_ids = tokenizer.encode(filler_raw, add_special_tokens=False)
    # 在 ID 列表层面复制 50 次，绝对不会触发 Tokenizer 长度警告
    filler_ids_pool = base_filler_ids * 50 
    
    # 3. 获取特殊字符 ID
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    
    # 4. 计算预算
    # 结构: [CLS] + Premise + [SEP] + Hypothesis + [SEP]
    # 固定开销: CLS(1) + SEP(1) + SEP(1) + Hyp + Facts
    fixed_len = 3 + len(hyp_ids) + len(f0_ids) + len(f1_ids) + len(f2_ids)
    
    filler_budget = max_len - fixed_len
    if filler_budget < 0: filler_budget = 0
    
    # 截取 Filler
    current_filler = filler_ids_pool[:filler_budget]
    
    # 5. 组装 Premise 部分
    if mode == "clumped":
        # Clumped at END: [Filler] + [F0 F1 F2]
        premise_ids = current_filler + f0_ids + f1_ids + f2_ids
    else:
        # Scattered: [F0] + [Filler/2] + [F1] + [Filler/2] + [F2]
        half = len(current_filler) // 2
        chunk1 = current_filler[:half]
        chunk2 = current_filler[half:]
        premise_ids = f0_ids + chunk1 + f1_ids + chunk2 + f2_ids
        
    # 6. 最终拼接
    input_ids_list = [cls_id] + premise_ids + [sep_id] + hyp_ids + [sep_id]
    
    # 7. 双重保险截断 (理论上不需要，但为了由 513 变 512 的边界情况)
    if len(input_ids_list) > max_len:
        input_ids_list = input_ids_list[:max_len]
        # 确保最后一个是 SEP
        input_ids_list[-1] = sep_id
        
    # 转 Tensor
    input_ids_tensor = torch.tensor([input_ids_list]).to(device)
    attention_mask = torch.ones_like(input_ids_tensor).to(device)
    
    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask
    }

# ==============================================================================
# 3. 运行对比
# ==============================================================================

def run_dense_comparison(samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    # 这里设置 model_max_length 为了防止 tokenizer 打印烦人的 warning，实际长度由代码控制
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    modes = ["clumped", "scattered"]
    
    print("\n" + "="*100)
    print(f"CONTROLLED EXPERIMENT: Dense Retrieval (Length/Width/Height)")
    print(f"Comparing Clumped (End) vs Scattered (Start/Mid/End)")
    print("="*100)
    
    results = {
        "clumped": {"correct": 0, "total": 0},
        "scattered": {"correct": 0, "total": 0, "pos_stats": {"Start":0, "Middle":0, "End":0, "Start_C":0, "Middle_C":0, "End_C":0}}
    }
    
    cases = [get_dense_case() for _ in range(samples)]
    
    for mode in modes:
        for case in cases:
            # 使用 Precise 版本构建输入
            inputs_dict = build_inputs_precise(tokenizer, case, mode=mode, max_len=512, device=device)
            
            with torch.no_grad():
                outputs = model(**inputs_dict)
            
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = model.config.id2label[pred_id]
            is_correct = (pred_label.lower() == case['expected'].lower())
            
            results[mode]['total'] += 1
            if is_correct: results[mode]['correct'] += 1
            
            if mode == "scattered":
                pos = case['target_pos']
                results['scattered']['pos_stats'][pos] += 1
                if is_correct: results['scattered']['pos_stats'][f"{pos}_C"] += 1

    print("\n" + "="*80)
    print("📊 Final Results (No Truncation Errors)")
    print("="*80)
    
    acc_clumped = results['clumped']['correct'] / results['clumped']['total'] * 100
    acc_scattered = results['scattered']['correct'] / results['scattered']['total'] * 100
    
    print(f"1. Clumped (All at End):  {acc_clumped:.1f}%")
    print(f"2. Scattered (Spread):    {acc_scattered:.1f}%")
    
    print("-" * 40)
    print("📍 Scattered Positional Breakdown:")
    s = results['scattered']['pos_stats']
    for pos in ["Start", "Middle", "End"]:
        tot = s[pos]
        corr = s[f"{pos}_C"]
        p = corr/tot*100 if tot>0 else 0
        print(f"   - Target at {pos:<6}: {p:.1f}% ({corr}/{tot})")

if __name__ == "__main__":
    run_dense_comparison(samples=300)