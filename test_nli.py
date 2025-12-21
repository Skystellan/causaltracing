import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import string

# -----------------------------------------------------------------------------
# 1. 随机数据生成器 (Data Generators)
# -----------------------------------------------------------------------------

def generate_typo(word):
    """随机在一个单词里替换/删除一个字符，模拟 Shaitan -> Shaitaan/Shiaton"""
    if len(word) < 3: return word
    idx = random.randint(1, len(word)-2)
    # 50% 概率替换，50% 概率重复字符
    if random.random() > 0.5:
        return word[:idx] + random.choice(string.ascii_lowercase) + word[idx+1:]
    else:
        return word[:idx] + word[idx] + word[idx:] # 重复字符 (taan)

def get_comparison_case():
    """生成数值比较用例 (静态关系)"""
    a = random.randint(10, 90)
    b = a - random.randint(1, 9) # b 肯定比 a 小
    
    # Premise: John has 50 apples.
    # Hypothesis: John has more than 45 apples.
    return {
        "type": "Comparison (Static)",
        "premise": f"According to the records, John has exactly {a} apples.",
        "hypothesis": f"According to the records, John has more than {b} apples.",
        "expected": "Entailment"
    }

def get_addition_case():
    """生成加法用例 (动态计算)"""
    # 避免 simple addition like 1+1, use larger numbers
    a = random.randint(10, 50)
    b = random.randint(10, 50)
    total = a + b
    wrong_total = total + random.choice([-1, 1])
    
    # Premise: Group A has 20. Group B has 30.
    # Hypothesis: Total is 51. (Should be Contradiction)
    return {
        "type": "Addition (Dynamic)",
        "premise": f"Group A collected {a} points, and Group B collected {b} points.",
        "hypothesis": f"In total, the groups collected {wrong_total} points.",
        "expected": "Contradiction" # 如果模型不会算，可能会判 Neutral 或 Entailment
    }

def get_counting_case():
    """生成计数用例 (枚举)"""
    names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"]
    selected = random.sample(names, k=random.randint(3, 5))
    count = len(selected)
    wrong_count = count - 1
    
    names_str = ", ".join(selected)
    
    # Premise: Alice, Bob, Charlie came.
    # Hypothesis: 2 people came.
    return {
        "type": "Counting (Enumeration)",
        "premise": f"The attendees included {names_str}.",
        "hypothesis": f"There were exactly {wrong_count} attendees.",
        "expected": "Contradiction"
    }

def get_entity_fluidity_case():
    """生成实体拼写变体用例 (文本流动性)"""
    base_names = ["Koechlin", "Shaitan", "Nambiar", "Khandelwal", "Devaiya"]
    target = random.choice(base_names)
    distorted = generate_typo(target)
    
    return {
        "type": "Text Fluidity",
        "premise": f"The critic praised the performance of {target} in the movie.",
        "hypothesis": f"The critic praised the performance of {distorted} in the movie.",
        "expected": "Entailment" # 我们期望模型容忍拼写错误
    }

# -----------------------------------------------------------------------------
# 2. 批量测试主程序
# -----------------------------------------------------------------------------

def run_batch_evaluation(num_samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # 统计器
    stats = {
        "Comparison (Static)": {"correct": 0, "total": 0},
        "Addition (Dynamic)": {"correct": 0, "total": 0},
        "Counting (Enumeration)": {"correct": 0, "total": 0},
        "Text Fluidity": {"correct": 0, "total": 0}
    }

    print(f"\nRunning batch evaluation on {device} with {num_samples} samples per category...\n")
    print(f"{'Type':<25} | {'Premise (Snippet)':<40} | {'Hypothesis (Snippet)':<30} | {'Exp':<8} | {'Pred':<8} | {'Result'}")
    print("-" * 140)

    # 生成所有测试用例
    all_cases = []
    for _ in range(num_samples):
        all_cases.append(get_comparison_case())
        all_cases.append(get_addition_case())
        all_cases.append(get_counting_case())
        all_cases.append(get_entity_fluidity_case())
    
    random.shuffle(all_cases) # 打乱顺序

    for case in all_cases:
        inputs = tokenizer(case['premise'], case['hypothesis'], truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        pred_label = model.config.id2label[pred_id]
        
        # 简化显示逻辑：DeBERTa 输出通常是 lowercase
        is_correct = (pred_label.lower() == case['expected'].lower())
        
        # 记录统计
        cat = case['type']
        stats[cat]['total'] += 1
        if is_correct:
            stats[cat]['correct'] += 1
            
        # 打印部分样例 (防止刷屏，每种类型只打前3个)
        if stats[cat]['total'] <= 3:
            p_show = case['premise'][:35] + "..."
            h_show = case['hypothesis'][:25] + "..."
            res_icon = "✅" if is_correct else "❌"
            print(f"{cat:<25} | {p_show:<40} | {h_show:<30} | {case['expected'][:4]:<8} | {pred_label[:4]:<8} | {res_icon}")

    print("-" * 140)
    print("\n📊 Final Accuracy Report:")
    for cat, data in stats.items():
        acc = (data['correct'] / data['total']) * 100
        print(f"{cat:<25}: {acc:.1f}% ({data['correct']}/{data['total']})")

    print("\n💡 Interpretation Guide:")
    print("1. Comparison > 90%? -> Model understands NUMBER MAGNITUDE (Semantic).")
    print("2. Addition < 50%?   -> Model fails ARITHMETIC (Calculation).")
    print("3. Fluidity > 90%?   -> Model tolerates TYPOS (Robustness).")

if __name__ == "__main__":
    run_batch_evaluation(num_samples=25) # 每种类型跑25个，总共100个