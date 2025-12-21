from transformers import AutoTokenizer

model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 刚才为你修改的 Premise
premise_text = """Michael Jordan was an American professional basketball player, widely regarded as the greatest player in the history of NBA. He was born on February 17, 1963, in Brooklyn, New York, and began playing basketball at a young age. He made his NBA debut in 1984 with the Chicago Bulls and quickly established himself as a spectacular player, known for his incredible athleticism and scoring ability.
Jordan played the majority of his career with the Chicago Bulls, earning 14 All-Star selections and 9 All-Defensive First Team selections. He won ten NBA scoring titles, was named the league's Most Valuable Player five times, and helped lead the Bulls to two separate "three-peat" championships, for a total of six NBA championships from 1991-1993 and 1996-1998.
Off the court, Jordan was known for his immense global influence and business acumen. He was deeply committed to elevating the athlete's role into a global cultural icon, profoundly changing sports marketing through his personal brand and endorsements. In recognition of his outstanding career and his impact on the sport, Jordan was inducted into the Naismith Memorial Basketball Hall of Fame in 2009. He remains a revered figure in the world of basketball and beyond, remembered for his unparalleled skill, his intense competitive drive, and his global influence."""

# 刚才为你修改的 Hypothesis
hypothesis_text = premise_text.replace("14 All-Star", "15 All-Star").replace("9 All-Defensive", "8 All-Defensive")

# 构建输入
inputs = tokenizer(premise_text, hypothesis_text)
total_tokens = len(inputs['input_ids'])

print(f"Token Count: {total_tokens}")

if total_tokens <= 512:
    print("✅ 完美！在窗口范围内，不会截断。")
    print("这是测试长文本注意力衰减的最佳样本。")
else:
    print("⚠️ 依然过长，需要进一步删减。")