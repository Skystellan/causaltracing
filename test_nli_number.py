import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# ==============================================================================
# 1. 辅助函数：生成各种复杂的数字逻辑样本
# ==============================================================================

def get_interval_case():
    """测试区间逻辑: 25岁 -> 20多岁"""
    age = random.randint(20, 59)
    decade = (age // 10) * 10
    wrong_decade = decade + 10
    
    # 50% 生成 Entailment (25 -> 20s), 50% 生成 Contradiction (25 -> 30s)
    if random.random() > 0.5:
        return {
            "type": "1. Interval Logic (Age)",
            "premise": f"The suspect is currently {age} years old.",
            "hypothesis": f"The suspect is in his {decade}s.",
            "expected": "Entailment"
        }
    else:
        return {
            "type": "1. Interval Logic (Age)",
            "premise": f"The suspect is currently {age} years old.",
            "hypothesis": f"The suspect is in his {wrong_decade}s.",
            "expected": "Contradiction"
        }

def get_decimal_precision_case():
    """测试高精度小数: 3.14159 vs 3.14158"""
    # 生成 5 位小数
    base = round(random.uniform(1, 10), 5)
    # 微调最后一位
    distorted = base + 0.00001
    
    return {
        "type": "2. High Precision Decimals",
        "premise": f"The scientific measurement was recorded precisely at {base:.5f} units.",
        "hypothesis": f"The scientific measurement was recorded precisely at {distorted:.5f} units.",
        "expected": "Contradiction"
    }

def get_large_number_format_case():
    """测试大数格式: 2,500,000 vs 2.5 million"""
    val = random.randint(1, 9)
    half = random.choice([0, 5])
    # 例如 2.5 million = 2,500,000
    num_str = f"{val},{half}00,000"
    word_str = f"{val}.{half} million"
    
    # 干扰项
    wrong_word_str = f"{val + 1}.{half} million"

    if random.random() > 0.5:
        return {
            "type": "3. Large Number Formats",
            "premise": f"The company profit hit {num_str} dollars last year.",
            "hypothesis": f"The company profit hit {word_str} dollars last year.",
            "expected": "Entailment"
        }
    else:
        return {
            "type": "3. Large Number Formats",
            "premise": f"The company profit hit {num_str} dollars last year.",
            "hypothesis": f"The company profit hit {wrong_word_str} dollars last year.",
            "expected": "Contradiction"
        }

def get_dense_info_case():
    """测试密集信息: A=10, B=20, C=30. 改 B"""
    metrics = ["Length", "Width", "Height", "Weight", "Depth"]
    random.shuffle(metrics)
    vals = [random.randint(10, 50) for _ in range(3)]
    
    # 构建 Premise: "Length is 10, Width is 20, Height is 30."
    premise_parts = [f"{metrics[i]} is {vals[i]}" for i in range(3)]
    premise = ", ".join(premise_parts) + "."
    
    # 选择中间的一个来改
    target_idx = 1 
    target_metric = metrics[target_idx]
    original_val = vals[target_idx]
    
    if random.random() > 0.5:
        # Entailment
        hypothesis = f"{target_metric} is {original_val}."
        expected = "Entailment"
    else:
        # Contradiction
        hypothesis = f"{target_metric} is {original_val + 1}."
        expected = "Contradiction"
        
    return {
        "type": "4. Dense Sentence Retrieval",
        "premise": f"Measurement details: {premise}",
        "hypothesis": f"Measurement details: {hypothesis}",
        "expected": expected
    }

def get_inequality_case():
    """测试不等式: 95 > 90"""
    score = random.randint(80, 99)
    threshold = score - random.randint(5, 10) # threshold 肯定比 score 小
    
    # 95 is more than 85 -> Entailment
    return {
        "type": "5. Inequality Logic (>)",
        "premise": f"The student achieved a score of {score} in the exam.",
        "hypothesis": f"The student achieved a score of more than {threshold} in the exam.",
        "expected": "Entailment"
    }

def get_chinese_numeral_case():
    """
    测试中文数字/不同语言数字 (Feature Check)
    注意：DeBERTa-v3 主要是英文模型，但也见过中文。
    这是一个 'Stress Test'。
    """
    map_cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
    val = random.randint(1, 10)
    cn_val = map_cn[val]
    
    # Premise: 英文数字
    # Hypothesis: 中文数字
    return {
        "type": "6. Cross-Lingual (Chinese)",
        "premise": f"There are {val} apples on the table.",
        "hypothesis": f"There are {cn_val} apples on the table.",
        "expected": "Entailment" # 看看模型能不能跨语言对齐
    }

# ==============================================================================
# 2. 执行引擎
# ==============================================================================

def run_comprehensive_validation(samples_per_type=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # 注册所有生成器
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
    print(f"{'Type':<30} | {'Premise (Snippet)':<35} | {'Hypothesis (Snippet)':<30} | {'Exp':<6} | {'Pred':<6} | {'Result'}")
    print("="*120)

    for gen_func in generators:
        # 预运行一次获取类型名称
        temp_case = gen_func()
        type_name = temp_case['type']
        stats[type_name] = {"correct": 0, "total": 0}
        
        for _ in range(samples_per_type):
            case = gen_func()
            
            inputs = tokenizer(case['premise'], case['hypothesis'], truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = model.config.id2label[pred_id]
            
            # 判定 (忽略大小写)
            is_correct = (pred_label.lower() == case['expected'].lower())
            
            stats[type_name]['total'] += 1
            if is_correct:
                stats[type_name]['correct'] += 1
            
            # 只打印前 2 个作为示例
            if stats[type_name]['total'] <= 2:
                icon = "✅" if is_correct else "❌"
                print(f"\n--- {type_name} ---")
                print(f"Premise: {case['premise']}")
                print(f"Hypothesis: {case['hypothesis']}")
                print(f"Expected: {case['expected']}")
                print(f"Predicted: {pred_label}")
                print(f"Result: {icon}")
                print("-" * 50)

    # ==============================================================================
    # 3. 统计报告
    # ==============================================================================
    print("\n" + "="*60)
    print("📊 Final Robustness Report")
    print("="*60)
    for type_name, data in stats.items():
        acc = (data['correct'] / data['total']) * 100
        print(f"{type_name:<30}: {acc:.1f}%  ({data['correct']}/{data['total']})")
    print("="*60)
    
    print("\n💡 Analysis:")
    print("1. If 'High Precision Decimals' > 90%: The model tokenizes numbers perfectly.")
    print("2. If 'Interval Logic' > 90%: The model understands numerical ranges (Semantics).")
    print("3. If 'Large Number Formats' > 90%: The model aligns '2.5 million' with '2,500,000'.")
    print("4. 'Cross-Lingual (Chinese)' is a stress test. Failure here is expected but success is impressive.")

if __name__ == "__main__":
    run_comprehensive_validation(samples_per_type=20) # 每类测20个，共120个