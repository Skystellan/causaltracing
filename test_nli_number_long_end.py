import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# ==============================================================================
# 1. 核心逻辑生成器 (保持原封不动)
# ==============================================================================

def get_interval_case():
    age = random.randint(20, 59)
    decade = (age // 10) * 10
    wrong_decade = decade + 10
    if random.random() > 0.5:
        return {"type": "1. Interval Logic (Age)", "premise": f"The suspect is currently {age} years old.", "hypothesis": f"The suspect is in his {decade}s.", "expected": "Entailment"}
    else:
        return {"type": "1. Interval Logic (Age)", "premise": f"The suspect is currently {age} years old.", "hypothesis": f"The suspect is in his {wrong_decade}s.", "expected": "Contradiction"}

def get_decimal_precision_case():
    base = round(random.uniform(1, 10), 5)
    distorted = base + 0.00001
    return {"type": "2. High Precision Decimals", "premise": f"The scientific measurement was recorded precisely at {base:.5f} units.", "hypothesis": f"The scientific measurement was recorded precisely at {distorted:.5f} units.", "expected": "Contradiction"}

def get_large_number_format_case():
    val = random.randint(1, 9)
    half = random.choice([0, 5])
    num_str = f"{val},{half}00,000"
    word_str = f"{val}.{half} million"
    wrong_word_str = f"{val + 1}.{half} million"
    if random.random() > 0.5:
        return {"type": "3. Large Number Formats", "premise": f"The company profit hit {num_str} dollars last year.", "hypothesis": f"The company profit hit {word_str} dollars last year.", "expected": "Entailment"}
    else:
        return {"type": "3. Large Number Formats", "premise": f"The company profit hit {num_str} dollars last year.", "hypothesis": f"The company profit hit {wrong_word_str} dollars last year.", "expected": "Contradiction"}

def get_dense_info_case():
    metrics = ["Length", "Width", "Height", "Weight", "Depth"]
    random.shuffle(metrics)
    vals = [random.randint(10, 50) for _ in range(3)]
    premise_parts = [f"{metrics[i]} is {vals[i]}" for i in range(3)]
    premise = ", ".join(premise_parts) + "."
    target_idx = 1 
    target_metric = metrics[target_idx]
    original_val = vals[target_idx]
    if random.random() > 0.5:
        return {"type": "4. Dense Sentence Retrieval", "premise": f"Measurement details: {premise}", "hypothesis": f"Measurement details: {target_metric} is {original_val}.", "expected": "Entailment"}
    else:
        return {"type": "4. Dense Sentence Retrieval", "premise": f"Measurement details: {premise}", "hypothesis": f"Measurement details: {target_metric} is {original_val + 1}.", "expected": "Contradiction"}

def get_inequality_case():
    score = random.randint(80, 99)
    threshold = score - random.randint(5, 10)
    return {"type": "5. Inequality Logic (>)", "premise": f"The student achieved a score of {score} in the exam.", "hypothesis": f"The student achieved a score of more than {threshold} in the exam.", "expected": "Entailment"}

def get_chinese_numeral_case():
    map_cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
    val = random.randint(1, 10)
    cn_val = map_cn[val]
    return {"type": "6. Cross-Lingual (Chinese)", "premise": f"There are {val} apples on the table.", "hypothesis": f"There are {cn_val} apples on the table.", "expected": "Entailment"}

# ==============================================================================
# 2. 上下文填充引擎 (Context Injection Engine)
# ==============================================================================

def inject_context(tokenizer, core_case, target_total_length=512):
    """
    将 core_case 中的短 premise 包装到长文本中。
    我们把核心事实放在 Filler 文本的【末尾】，这通常是最考验模型抗干扰能力的位置。
    """
    short_premise = core_case['premise']
    hypothesis = core_case['hypothesis']
    
    # 1. 准备一段足够长的 Filler Text (无关背景)
    # 这段文本是关于数据科学的废话，没有任何数字，旨在稀释注意力
    filler_base = (
        "In the modern era of big data and artificial intelligence, processing vast amounts of unstructured text "
        "has become a critical challenge for neural networks. Traditional models often struggle with long-range dependencies, "
        "forgetting information that appeared early in the sequence. Transformers have revolutionized this field "
        "by utilizing self-attention mechanisms that allow the model to weigh the importance of different tokens "
        "regardless of their positional distance. However, even with these advancements, distinguishing subtle details "
        "embedded within a sea of irrelevant context remains a stress test for model robustness. "
        "Researchers continuously develop new architectures like sparse attention and linear complexity models "
        "to handle context windows extending beyond thousands of tokens. Data quality, preprocessing, and "
        "fine-tuning strategies are equally important. "
    )
    long_filler = filler_base * 8 # 重复多次以确保足够长
    
    # 2. 计算预算
    # DeBERTa 输入结构: [CLS] Premise [SEP] Hypothesis [SEP]
    # 我们需要确保总长度不超 512，否则 core fact 可能会被截断
    
    # 先把核心部分编码
    hyp_ids = tokenizer.encode(hypothesis, add_special_tokens=False)
    core_prem_ids = tokenizer.encode(short_premise, add_special_tokens=False)
    
    # 预留特殊 token 的位置 (CLS, SEP, SEP) 约 3-4 个
    special_tokens_count = 4 
    
    # 计算 Filler 最多能有多少个 token
    max_filler_tokens = target_total_length - len(hyp_ids) - len(core_prem_ids) - special_tokens_count
    
    if max_filler_tokens <= 0:
        return short_premise # 核心太长了，无法填充
    
    # 3. 截取 Filler
    filler_ids = tokenizer.encode(long_filler, add_special_tokens=False)
    # 取前 max_filler_tokens 个
    selected_filler_ids = filler_ids[:max_filler_tokens]
    selected_filler_text = tokenizer.decode(selected_filler_ids, skip_special_tokens=True)
    
    # 4. 组装长 Premise
    # 格式: [无关废话] + [关键词提示] + [核心事实]
    # 这样核心事实会出现在大约第 400-500 个 token 的位置
    long_premise = f"{selected_filler_text} [KEY RECORD]: {short_premise}"
    
    return long_premise

# ==============================================================================
# 3. 批量压力测试主程序
# ==============================================================================

def run_context_stress_test(samples_per_type=20):
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # 2. 注册生成器
    generators = [
        get_interval_case,
        get_decimal_precision_case,
        get_large_number_format_case,
        get_dense_info_case,
        get_inequality_case,
        get_chinese_numeral_case
    ]
    
    stats = {}
    
    print("\n" + "="*120)
    print(f"{'Type':<30} | {'Tokens':<6} | {'Core Premise (Buried at End)':<35} | {'Hypothesis':<25} | {'Exp':<5} | {'Pred':<5}")
    print("="*120)

    for gen_func in generators:
        # 初始化统计
        temp = gen_func()
        t_name = temp['type']
        stats[t_name] = {"correct": 0, "total": 0}
        
        for _ in range(samples_per_type):
            # A. 获取短核心逻辑
            core_case = gen_func()
            
            # B. 注入长上下文 (核心步骤)
            # 我们请求填满 512 的窗口
            long_premise = inject_context(tokenizer, core_case, target_total_length=512)
            
            # C. 编码与验证长度
            inputs = tokenizer(
                long_premise, 
                core_case['hypothesis'], 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(device)
            
            actual_tokens = inputs['input_ids'].shape[1]
            
            # D. 推理
            with torch.no_grad():
                outputs = model(**inputs)
            
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = model.config.id2label[pred_id]
            
            # E. 判定
            is_correct = (pred_label.lower() == core_case['expected'].lower())
            
            stats[t_name]['total'] += 1
            if is_correct: stats[t_name]['correct'] += 1
            
            # 打印前 2 个作为示例，证明确实是长文本且核心在末尾
            if stats[t_name]['total'] <= 2:
                # 只打印 Premise 的最后 40 个字符，证明前面都是废话，核心在最后
                p_snippet = "..." + long_premise[-40:] 
                h_snippet = core_case['hypothesis'][:25]
                icon = "✅" if is_correct else "❌"
                print(f"{t_name:<30} | {actual_tokens:<6} | {p_snippet:<35} | {h_snippet:<25} | {core_case['expected'][:4]:<5} | {pred_label[:4]:<5} {icon}")

    # ==============================================================================
    # 4. 最终报告
    # ==============================================================================
    print("\n" + "="*80)
    print("📊 Final Robustness Report (Long Context Window ~512 Tokens)")
    print("="*80)
    
    all_passed = True
    for t_name, data in stats.items():
        acc = (data['correct'] / data['total']) * 100
        print(f"{t_name:<30}: {acc:.1f}% ({data['correct']}/{data['total']})")
        if acc < 90: all_passed = False
            
    print("="*80)
    print("\n💡 Conclusion:")
    if all_passed:
        print(">> STRONG EVIDENCE: The model maintains STATIC SEMANTIC capabilities (Intervals, Precision, Formats)")
        print("   even when the information is buried at the end of a full 512-token window.")
        print("   This confirms that 'Needle-in-a-Haystack' retrieval is solved for DeBERTa-v3.")
    else:
        print(">> MIXED RESULTS: Some capabilities degraded under long context.")

if __name__ == "__main__":
    # 每种类型测 20 个样本，总计 120 次推理
    run_context_stress_test(samples_per_type=20)