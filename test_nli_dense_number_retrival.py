import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# ==============================================================================
# 1. 核心数据生成器 (保持不变)
# ==============================================================================

def get_shuffled_profile_case():
    code = random.randint(100, 999)
    wrong_code = code + 1
    item_id = {"text": f"Subject ID is {code}.", "wrong_text": f"Subject ID is {wrong_code}.", "type": "ID"}
    
    depts = ["Logistics", "Research", "Security", "Finance", "Command"]
    dept = random.choice(depts)
    wrong_dept = "HR" if dept != "HR" else "Sales"
    item_dept = {"text": f"Department assignment is {dept}.", "wrong_text": f"Department assignment is {wrong_dept}.", "type": "Dept"}
    
    level = random.randint(1, 9)
    wrong_level = level + 1
    item_level = {"text": f"Clearance level is {level}.", "wrong_text": f"Clearance level is {wrong_level}.", "type": "Level"}
    
    all_items = [item_id, item_dept, item_level]
    random.shuffle(all_items)
    
    target_idx = random.choice([0, 1, 2])
    target_item = all_items[target_idx]
    pos_map = {0: "Start", 1: "Middle", 2: "End"}
    
    if random.random() > 0.5:
        hyp = target_item['text']
        expected = "Entailment"
    else:
        hyp = target_item['wrong_text']
        expected = "Contradiction"
        
    return {
        "ordered_facts": [item['text'] for item in all_items],
        "hypothesis": hyp,
        "expected": expected,
        "target_pos": pos_map[target_idx],
        "target_type": target_item['type']
    }

# ==============================================================================
# 2. 动态 ID 组装引擎 (修复了 Tokenizer 报警的问题)
# ==============================================================================

def build_inputs_from_ids(tokenizer, case, mode="clumped", max_len=512, device="cpu"):
    # 1. 获取基础 ID
    facts = case['ordered_facts']
    hyp_ids = tokenizer.encode(case['hypothesis'], add_special_tokens=False)
    fact0_ids = tokenizer.encode(facts[0], add_special_tokens=False)
    fact1_ids = tokenizer.encode(facts[1], add_special_tokens=False)
    fact2_ids = tokenizer.encode(facts[2], add_special_tokens=False)
    
    # --- 修复点：安全生成 Filler ID Pool ---
    single_filler_text = (
        "In information theory, data compression involves encoding information using fewer bits than the original representation. "
        "Encryption is the process of encoding information. "
    )
    # 先把这一小段转成 ID (这绝对不会超 512)
    base_filler_ids = tokenizer.encode(single_filler_text, add_special_tokens=False)
    
    # 然后在 ID 列表层面复制 20 次，生成足够长的池子
    # 这样 Tokenizer 永远不会看到长文本，也就不会报错
    filler_ids_pool = base_filler_ids * 20 
    
    # 2. 预算计算
    # [CLS] + Premise + [SEP] + Hypothesis + [SEP]
    # 3 special tokens
    core_len = len(fact0_ids) + len(fact1_ids) + len(fact2_ids)
    reserved = 3 + len(hyp_ids)
    
    filler_budget = max_len - reserved - core_len
    if filler_budget < 0: filler_budget = 0
    
    # 3. 截取 Filler
    current_filler_ids = filler_ids_pool[:filler_budget]
    
    # 4. 组装 Premise ID
    if mode == "clumped":
        premise_ids = current_filler_ids + fact0_ids + fact1_ids + fact2_ids
    else:
        half = len(current_filler_ids) // 2
        chunk1 = current_filler_ids[:half]
        chunk2 = current_filler_ids[half:]
        premise_ids = fact0_ids + chunk1 + fact1_ids + chunk2 + fact2_ids
        
    # 5. 最终组装
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    input_ids_list = [cls_id] + premise_ids + [sep_id] + hyp_ids + [sep_id]
    
    # 6. 最终安全截断 (双重保险)
    if len(input_ids_list) > max_len:
        input_ids_list = input_ids_list[:max_len]
        # 确保最后一个是 SEP (如果被截断了的话)
        input_ids_list[-1] = sep_id
        
    # 7. 转 Tensor
    input_ids_tensor = torch.tensor([input_ids_list]).to(device)
    attention_mask = torch.ones_like(input_ids_tensor).to(device)
    
    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask
    }

# ==============================================================================
# 3. 测试运行
# ==============================================================================

def run_debugged_test(samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 显式关闭 tokenizer 的最大长度警告，虽然上面的修复已经从根本上解决了
    tokenizer.model_max_length = 100000 
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    modes = ["clumped", "scattered"]
    
    print("\n" + "="*120)
    print(f"{'Mode':<10} | {'Pos':<6} | {'Tokens':<6} | {'Hypothesis':<25} | {'Exp':<5} | {'Pred':<5} | {'Res'}")
    print("="*120)

    results = {
        "clumped": {"correct": 0, "total": 0},
        "scattered": {"correct": 0, "total": 0, "pos_stats": {"Start":0, "Middle":0, "End":0, "Start_C":0, "Middle_C":0, "End_C":0}}
    }
    
    core_cases = [get_shuffled_profile_case() for _ in range(samples)]
    
    for mode in modes:
        for case in core_cases:
            # 这里的 max_len 设为 512
            inputs = build_inputs_from_ids(tokenizer, case, mode=mode, max_len=512, device=device)
            
            # --- 调试：打印实际长度，证明没有超标 ---
            actual_len = inputs['input_ids'].shape[1]
            if actual_len > 512:
                print(f"❌ 严重错误：生成了 {actual_len} 长度的 Tensor！")
                break
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = model.config.id2label[pred_id]
            is_correct = (pred_label.lower() == case['expected'].lower())
            
            results[mode]['total'] += 1
            if is_correct: results[mode]['correct'] += 1
            
            if mode == "scattered":
                pos = case['target_pos']
                results['scattered']['pos_stats'][pos] += 1
                if is_correct: results['scattered']['pos_stats'][f"{pos}_C"] += 1

            if mode == "scattered" and results[mode]['total'] <= 5:
                icon = "✅" if is_correct else "❌"
                print(f"{mode:<10} | {case['target_pos']:<6} | {actual_len:<6} | {case['hypothesis'][:25]:<25} | {case['expected'][:4]:<5} | {pred_label[:4]:<5} | {icon}")

    print("\n" + "="*80)
    print("📊 Final Clean Report")
    print("="*80)
    
    acc_clumped = (results['clumped']['correct'] / results['clumped']['total']) * 100
    acc_scattered = (results['scattered']['correct'] / results['scattered']['total']) * 100
    
    print(f"1. Clumped (All at End):  {acc_clumped:.1f}%")
    print(f"2. Scattered (Spread):    {acc_scattered:.1f}%")
    
    print("-" * 40)
    print("📍 Positional Breakdown:")
    s_stats = results['scattered']['pos_stats']
    for pos in ["Start", "Middle", "End"]:
        total = s_stats[pos]
        corr = s_stats[f"{pos}_C"]
        if total > 0:
            p_acc = (corr / total) * 100
            print(f"   - Target at {pos:<6}: {p_acc:.1f}% ({corr}/{total})")
        else:
            print(f"   - Target at {pos:<6}: N/A")

if __name__ == "__main__":
    run_debugged_test(samples=300)