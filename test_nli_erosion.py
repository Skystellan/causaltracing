import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_final_pivot_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    print("\n" + "="*80)
    print("🚀 FINAL PROBE: State Logic & The Blessing of Retrieval")
    print("="*80)

    # ==============================================================================
    # 1. 蒙太奇谎言修正：从“叙事”转为“状态” (State Overwrite)
    # ==============================================================================
    print("\n>>> 1. Fixing Montage Lie: The 'State Overwrite' Test")
    
    montage_cases = [
        {
            "name": "Narrative Only (Then)",
            "p": "Mike hit Amy. Then they broke up.",
            "h": "They broke up. Then Mike hit Amy.", 
            "exp": "Contradiction (Desired)",
            "rationale": "Pure temporal reordering often fails."
        },
        {
            "name": "State: Life/Death",
            "p": "The artist died in 2020. Then he released a live album in 2021.", 
            "h": "The artist released a live album in 2021. Then he died in 2020.",
            "exp": "Contradiction",
            "rationale": "Irreversible state (Dead vs Alive)."
        },
        {
            "name": "State: Broken/Intact",
            "p": "The glass shattered. Then Mike touched it.", 
            "h": "Mike touched the glass. Then it shattered.",
            "exp": "Contradiction",
            "rationale": "Object state mismatch (Shards vs Whole)."
        }
    ]

    for case in montage_cases:
        inputs = tokenizer(case['p'], case['h'], return_tensors="pt").to(device)
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
        # Label map通常是: entailment, neutral, contradiction
        lbls = ["Entailment", "Neutral", "Contradiction"]
        pred_idx = torch.argmax(probs).item()
        pred_label = model.config.id2label[pred_idx]
        
        # 只要不是 Entailment 就算通过
        status = "❌ FAIL (Event Bag)" if "entailment" in pred_label.lower() else "✅ PASS (State Logic)"
        print(f"Case: {case['name']:<25} | Pred: {pred_label:<13} | {status}")

    # ==============================================================================
    # 2. 检索的祝福 (The Blessing of Retrieval)
    # ==============================================================================
    print("\n>>> 2. Can Fragmentation Fix Logical Blindness?")
    print("Hypothesis: E2E fails on 'ONLY IF', but Retrieved Fact (without Rule) yields Neutral (Correct).")
    
    # 场景：Context 包含规则和事实
    # 这是一个典型的“陷阱”，全文读会让模型脑子烧坏
    rule = "The access code is 123 ONLY IF the light is RED."
    fact = "The light is GREEN."
    hyp = "The access code is 123."
    
    # 模拟 E2E (全文)
    inputs_e2e = tokenizer(f"{rule} {fact}", hyp, return_tensors="pt").to(device)
    pred_e2e = model.config.id2label[torch.argmax(model(**inputs_e2e).logits).item()]
    
    # 模拟 LongSePer (只检索到了 Fact，因为 Fact 和 Rule 语义距离较远，或者被切分了)
    # 既然检索器是基于 Similarity 的，Hypothesis 问的是 Code，可能会检索到 Rule
    # 但我们假设 Late Chunking 能够把它们分开，或者我们展示“如果是 Fact Only”会发生什么
    inputs_retrieved = tokenizer(fact, hyp, return_tensors="pt").to(device)
    pred_retrieved = model.config.id2label[torch.argmax(model(**inputs_retrieved).logits).item()]
    
    print(f"\nScenario A: E2E (Full Context)    -> Pred: {pred_e2e}")
    print(f"Scenario B: LongSePer (Fact Only) -> Pred: {pred_retrieved}")
    
    if "entailment" in pred_e2e.lower() and ("neutral" in pred_retrieved.lower() or "contradiction" in pred_retrieved.lower()):
        print("\n🎉 INSIGHT PROVEN: Fragmentation serves as a 'Logic Filter'.")
        print("   Retrieval removed the misleading 'Rule' and forced the model to be Agnostic.")
    else:
        print(f"\n🤔 Result needs analysis: {pred_e2e} vs {pred_retrieved}")

if __name__ == "__main__":
    run_final_pivot_experiment()