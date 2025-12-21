import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def count_tokens(text, tokenizer):
    """计算文本的token数量"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def run_simple_nli_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    # model_name = "microsoft/deberta-v2-xlarge-mnli"
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # 直接在脚本中定义输入对，修改这里即可
    test_cases = [
        {
            "premise": """The board released its quarterly report on recent strategic expansions. In early February, we successfully finalized the acquisition of AlphaTech, a promising startup in the edge computing space. Their innovative portfolio is expected to enhance our product line significantly. According to the finalized integration documents, AlphaTech brings approximately 235 employees into our engineering and R&D divisions.

The mid-quarter focus shifted to operational efficiency and market analysis. Several initiatives were launched to streamline our supply chain and assess competitive landscapes in the Asia-Pacific region, with findings to be presented next month.

Subsequently, in March, the merger with BetaSys was completed as per the agreement signed last fall. BetaSys's mature customer relationship platform is a key asset. The integration process has transferred a team of about 1,240 employees, primarily from sales and customer support, to our commercial operations unit.

Finally, just last week, we closed the deal to acquire GammaLab, a leader in data analytics solutions. This move strengthens our advisory service offerings. The acquisition adds roughly 625 data scientists and industry consultants to our talent pool, with full onboarding scheduled for next quarter.
""",
            "hypothesis": """The recent acquisitions have added approximately 2,100 new employees to the company."""
        },
        {
            "premise": """The City Parks and Recreation Department has published its review of major sporting events held this spring season, highlighting their economic and community impact.

The annual City International Marathon kicked off the season in April. It featured a highly competitive elite field and record participation from amateur runners. An economic impact study commissioned by the tourism board estimates that the marathon generated direct spending in local hospitality, retail, and services totaling about $3.2 million.

Following that, in mid-May, the National Junior Swimming Championships took place at the city's aquatic center over five days. The event showcased emerging talent and set several new national age-group records. While spectator turnout was moderate, the event still contributed an estimated $1.8 million in direct economic activity to the downtown area.

The season concluded with the widely popular Cycling Grand Prix, a challenging circuit race through urban and scenic routes. The event attracted large crowds along the route and significant media coverage. Preliminary figures from the chamber of commerce indicate a direct economic boost of approximately $2.5 million from visitor spending related to the race.
""",
            "hypothesis": """The direct economic impact from the city's major spring sporting events was around $7.5 million."""
        },
        {
            "premise": """
Project "Nexus" - Q3 Budget & Progress Review Meeting Minutes

Date: November 14
Attendees: Project Sponsor, Module Leads, Finance Controller

Discussion Summary:

Module Lead (Backend Infrastructure) reported: Development is complete, and the module is now in the optimization and stress-testing phase. Expenditure has remained tightly controlled. The total fiscal year-to-date spend for this module is $1,871,523, aligning perfectly with its allocated budget.

Module Lead (Frontend & UX) reported: The final design system has been approved, and development is 85% complete. Some unforeseen costs arose from licensing a specialized animation library. The current fiscal year expenditure for this module stands at $6,113,276. This is slightly above the initial forecast but is accounted for within the module's contingency allowance.

Module Lead (Data Integration Layer) reported: Data pipelines are operational and handling 90% of target load. Cloud service costs have been as projected. The cumulative spend for this module in the current fiscal year is $99,394.

Action Items: The finance controller will consolidate the figures from all modules and external vendor costs to prepare a comprehensive Q4 budget forecast for executive review.
""",
            "hypothesis": """The total combined expenditure for the three core modules (Backend Infrastructure, Frontend & UX, Data Integration Layer) of Project Nexus this fiscal year is $8,000,000."""
        },
        {
            "premise": """
Mike hit Amy. Mike and Amy broke up. Amy went to movies with John.
""",
            "hypothesis": """Amy went to movies with John. Mike hit Amy. Mike and Amy broke up."""
        },
        {
            "premise": """
The apple is expensive and the orange is not sweet.
""",
            "hypothesis": """The apple is not sweet."""
        },
        {
            "premise": """
A deer is jumping over a fence.
""",
            "hypothesis": """A deer is not not jumping over a fence."""
        },
        {
            "premise": """
Pablo Aurrecochea (born March 3, 1981 in Artigas, Uruguay) is a Uruguayan footballer...
""",
            "hypothesis": """Pablo Aurrecochea has been playing for 60 years."""
        },
        {
            "premise": """
Pablo Aurrecochea (born March 3, 1981 in Artigas, Uruguay) is a Uruguayan footballer...
""",
            "hypothesis": """Pablo Aurrecochea has been playing for 60 years."""
        },
        {
            "premise": """
5 percent probability that each part will be defect free.
""",
            "hypothesis": """Each part has a 95 percent chance of having a defect."""
        },
        {
            "premise": """15% of the 50 patients in the control group and 20% of the 50 patients in the active group reported side effects.""",
            "hypothesis": """More than 15 patients total experienced side effects.""",
        },
        {
            "premise": """Code is 123 ONLY IF Light is RED. Light is GREEN.""",
            "hypothesis": """The access code is 123.""",
        },
        {
            "premise": """
Mike hit Amy. Then Mike and Amy broke up. Then Amy went to movies with John.
""",
            "hypothesis": """Amy went to movies with John. Then Mike hit Amy. Then Mike and Amy broke up."""
        },
        {
            "premise": """
Mike hit Amy. Then Mike and Amy broke up. Then Amy went to movies with John.
""",
            "hypothesis": """Amy went to movies with John. Then Mike and Amy broke up. Then Mike hit Amy."""
        },
        {
            "premise": """
The glass shattered. Mike touched it.
""",
            "hypothesis": """Mike touched the glass. The glass shattered."""
        },
        {
            "premise": """
The glass shattered. Then Mike touched it.
""",
            "hypothesis": """Mike touched the glass. Then the glass shattered."""
        },
        {
            "premise": """She performed in Shaitan.""",
            "hypothesis": "She performed in Satan."
        },
        {
            "premise": """She performed in Satin.""",
            "hypothesis": "She performed in Satan."
        },
        {
            "premise": """She performed in Satan.""",
            "hypothesis": "She performed in Satan."
        },
        # 添加更多测试案例...
    ]
    
    print("Running NLI tests...")
    
    for i, case in enumerate(test_cases, 1):
        premise = case["premise"]
        hypothesis = case["hypothesis"]
        
        # 计算 token 数并检查
        prem_tokens = count_tokens(premise, tokenizer)
        hyp_tokens = count_tokens(hypothesis, tokenizer)
        total_tokens = prem_tokens + hyp_tokens + 3  # CLS, SEP, SEP
        
        if total_tokens > 512:
            print(f"⚠️  DEBUG: Test {i} exceeds context window!")
            print(f"   Premise tokens: {prem_tokens}")
            print(f"   Hypothesis tokens: {hyp_tokens}")
            print(f"   Total tokens: {total_tokens} (> 512)")
            print(f"   Premise: {premise[:100]}...")
            print(f"   Hypothesis: {hypothesis[:100]}...")
            print("   Note: Input will be truncated by tokenizer.")
        
        inputs = tokenizer(
            premise, 
            hypothesis, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=1)[0].cpu().tolist()
        label_map = model.config.id2label
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        pred_label = label_map[pred_id]
        
        s_ent = probs[model.config.label2id.get('entailment', model.config.label2id.get('ENTAILMENT', 2))]
        s_neu = probs[model.config.label2id.get('neutral', model.config.label2id.get('NEUTRAL', 1))]
        s_con = probs[model.config.label2id.get('contradiction', model.config.label2id.get('CONTRADICTION', 0))]
        
        print(f"Test {i}:")
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Predicted: {pred_label}")
        print(f"Entailment: {s_ent:.4f}, Neutral: {s_neu:.4f}, Contradiction: {s_con:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    run_simple_nli_test()