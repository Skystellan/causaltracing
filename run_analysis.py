import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from entailment_rome.entailment_model import EntailmentModelAndTokenizer
from entailment_experiments.causal_trace import plot_hidden_flow

def main():
    # 直接在代码中定义输入，避免命令行转义问题
    premise = """She performed in Satan."""
    hypothesis = """She performed in Satan"""
    premise_noise_tokens = ["Satan"]
    hypothesis_noise_tokens = []
    samples = 5
    noise = 0.1
    output_dir = "results/Satan_Satan"
    format_type = "png"
    target_label = "entailment"
    debug_hooks = False
    debug_tokens = False
    # model = "microsoft/deberta-v2-xlarge-mnli"
    model = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    # 如果需要命令行覆盖，可以取消注释下面的代码
    # parser = argparse.ArgumentParser(description="运行因果追踪分析并生成 PNG 图片")
    # ... (其余argparse代码保持不变，但注释掉)
    # args = parser.parse_args()
    # 然后用args.xxx替换上面的变量
    
    print(f"🚀 因果追踪分析 - {format_type.upper()} 格式")
    print("=" * 50)
    
    print(f"📥 Loading model: {model}")
    
    try:
        mt = EntailmentModelAndTokenizer(model)
        print(f"✅ Model loaded successfully!")
        print(f"   Model ID: {model}")
        print(f"   config.model_type: {mt.model.config.model_type}")
        print(f"   config.architectures: {getattr(mt.model.config, 'architectures', None)}")
        print(f"   config.id2label: {getattr(mt.model.config, 'id2label', None)}")
        print(f"   Layers: {mt.num_layers}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    print(f"\n🔍 分析配置:")
    print(f"   Premise: {premise}")
    print(f"   Hypothesis: {hypothesis}")
    print(f"   Premise Noise Tokens: {premise_noise_tokens}")
    print(f"   Hypothesis Noise Tokens: {hypothesis_noise_tokens}")
    print(f"   噪声样本: {samples}")
    print(f"   噪声水平: {noise}")
    print(f"   目标类别: {target_label}")
    print(f"   Debug hooks: {debug_hooks}")
    print(f"   Debug tokens: {debug_tokens}")
    print(f"   输出格式: {format_type.upper()}")
    
    # 创建输出目录
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 分析列表
    analyses = [
        ("All Components", None, "全组件分析"),
        ("MLP Components", "mlp", "MLP组件分析"), 
        ("Attention Components", "attn", "Attention组件分析")
    ]
    
    print(f"\n🎨 生成 {format_type.upper()} 热力图...")
    
    for analysis_name, kind, description in analyses:
        print(f"   📊 {analysis_name}...")
        
        try:
            if format_type == "png":
                output_file = output_dir_path / f"causal_trace_{kind or 'all'}.png"
                plot_hidden_flow(
                    mt=mt,
                    premise=premise,
                    hypothesis=hypothesis,
                    premise_noise_tokens=premise_noise_tokens,
                    hypothesis_noise_tokens=hypothesis_noise_tokens,
                    samples=samples,
                    noise=noise,
                    kind=kind,
                    target_label=target_label,
                    debug_hooks=debug_hooks,
                    debug_tokens=debug_tokens,
                    savepng=str(output_file)
                )
            else:  # PDF
                output_file = output_dir_path / f"causal_trace_{kind or 'all'}.pdf"
                plot_hidden_flow(
                    mt=mt,
                    premise=premise,
                    hypothesis=hypothesis,
                    premise_noise_tokens=premise_noise_tokens,
                    hypothesis_noise_tokens=hypothesis_noise_tokens,
                    samples=samples,
                    noise=noise,
                    kind=kind,
                    target_label=target_label,
                    debug_hooks=debug_hooks,
                    debug_tokens=debug_tokens,
                    savepdf=str(output_file)
                )
            
            file_size = output_file.stat().st_size / 1024  # KB
            print(f"   ✅ {description} 保存到: {output_file} ({file_size:.1f} KB)")
            
        except Exception as e:
            print(f"   ❌ {analysis_name} 生成失败: {e}")
            continue
    
    print(f"\n🎉 分析完成!")
    print(f"📁 所有文件保存在: {output_dir_path}")

if __name__ == "__main__":
    exit(main())