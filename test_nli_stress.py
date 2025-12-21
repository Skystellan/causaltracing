import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_logic_breakdown():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    # 使用你论文中的强力模型
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # 既然 v3 在 512 内视力无敌，我们就在它的舒适区(512)内攻击它的逻辑死穴
    MAX_LEN = 512
    
    print("\n" + "="*80)
    print("🧠 LOGIC BREAKDOWN PROBE: STATE MUTATION & CONDITIONAL LOGIC")
    print("Hypothesis: V3 sees all facts, but cannot handle 'Overwrite' or 'Condition'.")
    print("="*80)

    scenarios = ["Chronological_Overwrite", "Conditional_Trap"]
    
    for scenario in scenarios:
        correct = 0
        samples = 50 
        print(f"\n>>> Running Scenario: {scenario}")
        
        for _ in range(samples):
            # ==================================================================
            # 场景 A: ⏳ 时序覆盖 (Chronological Overwrite)
            # 难点：同一个实体有多个状态，模型必须识别“最新”的那个
            # ==================================================================
            if scenario == "Chronological_Overwrite":
                entity = f"Ticket-{random.randint(1000,9999)}"
                
                # 定义三个状态
                status_old = "Open"
                status_mid = "In-Progress"
                status_new = "Closed" # 真理 (Truth)
                
                # 构造日志流 (Log Stream)
                # 1. 早期日志 (Old - Distractor)
                log_1 = f"Timestamp 08:00: System auto-generated {entity} with status {status_old}."
                # 2. 中期日志 (Mid - Distractor)
                log_2 = f"Timestamp 12:00: Engineer updated {entity} to status {status_mid}."
                # 3. 晚期日志 (New - Ground Truth)
                log_3 = f"Timestamp 18:00: Final resolution applied. {entity} status is now {status_new}."
                
                # 填充噪音 (让模型无法仅靠距离判断，而是必须理解 update 逻辑)
                distractors = []
                for i in range(15):
                    d_id = f"Ticket-{random.randint(1000,9999)}"
                    d_stat = random.choice(["Open", "Closed", "Error"])
                    distractors.append(f"Log: {d_id} is {d_stat}.")
                
                # 拼装：Old -> Noise -> Mid -> Noise -> New
                # 所有的状态都在 Context 里，模型都能"看见"。
                split = len(distractors)//2
                context = f"{log_1} {' '.join(distractors[:split])} {log_2} {' '.join(distractors[split:])} {log_3}"
                
                # 攻击：拿着“旧状态”去问模型 (诱导它 Entailment)
                # 事实是 Closed，所以 Open 是 Contradiction
                if random.random() > 0.5:
                    hyp = f"{entity} is currently {status_old}."
                    expected = "Contradiction" 
                else:
                    hyp = f"{entity} is currently {status_new}."
                    expected = "Entailment"

            # ==================================================================
            # 场景 B: 🚦 条件陷阱 (Conditional Trap)
            # 难点：If A then B. 文中只有 A' (似是而非)。
            # ==================================================================
            elif scenario == "Conditional_Trap":
                code = random.randint(100, 999)
                target_color = "RED"
                
                # 规则：只有红灯亮，密码才是 code
                rule = f"Security Protocol: The access code is {code} ONLY IF the alert light is {target_color}."
                
                # 事实：绿灯亮了
                fact = f"System Report: Currently, the alert light is GREEN."
                
                # 干扰项：大量的 If...Then...
                distractors = []
                colors = ["BLUE", "YELLOW", "PURPLE"]
                for c in colors:
                    d_code = random.randint(100, 999)
                    distractors.append(f"Rule: Code is {d_code} if light is {c}.")
                
                context = f"{rule} {' '.join(distractors)} {fact}"
                
                # 攻击：问密码是不是 code
                # 模型看到了 "Code is {code}" (在规则里)，也看到了 "Light is GREEN" (在事实里)
                # 它需要做推理：RED != GREEN -> Condition False -> Code Not Valid
                hyp = f"The access code is {code}."
                expected = "Not_Entailment" # 应该是 Neutral 或 Contradiction，绝不是 Entailment

            # --- 编码与截断保护 ---
            inputs = tokenizer(context, hyp, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
            
            # 完整性自检：确保关键信息(新状态/条件)都在
            decoded = tokenizer.decode(inputs.input_ids[0])
            if scenario == "Chronological_Overwrite":
                if status_new not in decoded: continue 
            elif scenario == "Conditional_Trap":
                if "GREEN" not in decoded or str(code) not in decoded: continue

            # --- 推理 ---
            with torch.no_grad():
                out = model(**inputs)
            
            pred_label = model.config.id2label[torch.argmax(out.logits).item()]
            
            # 宽松判定 (Robust Evaluation)
            is_correct = False
            if expected == "Entailment":
                if "entailment" in pred_label.lower(): is_correct = True
            elif expected == "Contradiction":
                if "contradiction" in pred_label.lower(): is_correct = True
            elif expected == "Not_Entailment":
                # 只要不是 Entailment 就算对 (Neutral/Contradiction 都是合理的拒绝)
                if "entailment" not in pred_label.lower(): is_correct = True 
            
            if is_correct: correct += 1
        
        acc = correct / samples * 100
        print(f"Scenario Accuracy: {acc:.1f}%")
        
        if acc < 85:
            print(f"   >>> 💥 CRITICAL HIT: Model failed logic/state reasoning in {scenario}.")

if __name__ == "__main__":
    run_logic_breakdown()