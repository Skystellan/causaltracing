import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# ==============================================================================
# 1. 恶劣环境生成器
# ==============================================================================

def get_reproduction_case():
    # 核心事实
    target_val = random.randint(100, 999) # 3位数
    core_text = f"Subject ID is {target_val}."
    
    # 干扰项 (Distractors)
    # 关键点：让干扰项的数值也是 3位数，且结构极度相似
    # 这种同质化数据会让 Attention Head 很难区分 value 到底属于哪个 key
    distractor_templates = [
        "Zone Code is {}",
        "Batch Num is {}",
        "Area ID is {}",
        "Unit No is {}",
        "Item Key is {}"
    ]
    
    distractors = []
    for temp in distractor_templates:
        # 生成干扰数值 (避免和 target 一样)
        d_val = random.randint(100, 999)
        while d_val == target_val:
            d_val = random.randint(100, 999)
        distractors.append(temp.format(d_val))
    
    # 随机打乱 Core 和 Distractors 的顺序
    # 模拟一堆数据堆在一起的情况
    block_items = distractors + [core_text]
    random.shuffle(block_items)
    
    # Hypothesis
    if random.random() > 0.5:
        hyp = f"Subject ID is {target_val}."
        expected = "Entailment"
    else:
        hyp = f"Subject ID is {target_val + 1}."
        expected = "Contradiction"
        
    return {
        "block_items": block_items, # 这是一个列表，包含核心和干扰
        "hypothesis": hyp,
        "expected": expected
    }

# ==============================================================================
# 2. 注入引擎 (ID级精确组装，Start位置)
# ==============================================================================

def build_reproduction_input(tokenizer, case, max_len=512, device="cpu"):
    hyp_ids = tokenizer.encode(case['hypothesis'], add_special_tokens=False)
    
    # 将 Block 里的句子连成一片
    # "Zone Code is 111. Subject ID is 222. Area ID is 333..."
    block_text = " ".join(case['block_items'])
    block_ids = tokenizer.encode(block_text, add_special_tokens=False)
    
    # Filler
    filler_raw = "System check nominal. Data stream active. Monitoring parameters. "
    base_filler_ids = tokenizer.encode(filler_raw, add_special_tokens=False)
    filler_pool = base_filler_ids * 50
    
    # 预算
    reserved = 3 + len(hyp_ids) + len(block_ids)
    budget = max_len - reserved
    if budget < 0: budget = 0
    
    curr_filler = filler_pool[:budget]
    
    # 组装：[CLS] [Block (Core+Dist)] [Filler] [SEP] [Hyp] [SEP]
    # 这种结构让 Core 处于最远的 Start 位置，且周围全是干扰
    cls = [tokenizer.cls_token_id]
    sep = [tokenizer.sep_token_id]
    
    input_ids = cls + block_ids + curr_filler + sep + hyp_ids + sep
    
    # 安全截断
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        input_ids[-1] = sep[0]
        
    return torch.tensor([input_ids]).to(device)

# ==============================================================================
# 3. 运行测试
# ==============================================================================

def run_reproduction_attempt(samples=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    print("\n" + "="*100)
    print(f"REPRODUCTION ATTEMPT: Can we break it again?")
    print(f"Setup: 'Subject ID' mixed with 5 similar 'Key-Value' pairs at START.")
    print(f"All values are 3-digit numbers (Homogeneous Interference).")
    print("="*100)
    
    correct = 0
    for i in range(samples):
        case = get_reproduction_case()
        inputs = build_reproduction_input(tokenizer, case, max_len=512, device=device)
        
        with torch.no_grad():
            out = model(inputs)
        pred = model.config.id2label[torch.argmax(out.logits).item()]
        
        if pred.lower() == case['expected'].lower():
            correct += 1
            
        # 打印第一个 Case 看看长什么样
        if i == 0:
            decoded = tokenizer.decode(inputs[0])
            print(f"Sample Input (Truncated): {decoded[:150]} ... {decoded[-50:]}")
            
    acc = correct / samples * 100
    print("\n" + "="*80)
    print(f"Accuracy: {acc:.1f}%")
    print("="*80)
    
    if acc < 70:
        print(">> SUCCESS: We reproduced the collapse!")
        print(">> Reason: Homogeneous numeric binding at long distance is the weak point.")
    else:
        print(">> FAILED: The model is simply too robust.")
        print(">> Conclusion: The previous 48.5% might have been a truncation artifact or specific outlier.")

if __name__ == "__main__":
    run_reproduction_attempt(samples=100)