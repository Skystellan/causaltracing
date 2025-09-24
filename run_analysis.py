import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from entailment_rome.entailment_model import EntailmentModelAndTokenizer
from entailment_experiments.causal_trace import plot_hidden_flow

def main():
    parser = argparse.ArgumentParser(description="运行因果追踪分析并生成 PNG 图片")
    parser.add_argument("--model", default="microsoft/deberta-base-mnli", 
                       help="模型名称 (默认: deberta-base-mnli)")
    parser.add_argument("--premise", default="He was born in 1934.", 
                       help="前提句子")
    parser.add_argument("--hypothesis", default="He was born in 1935.", 
                       help="假设句子")
    parser.add_argument("--numerical-tokens", nargs="+", default=["1934", "1935"],
                       help="数值 tokens")
    parser.add_argument("--samples", type=int, default=10, 
                       help="噪声样本数量")
    parser.add_argument("--noise", type=float, default=0.1, 
                       help="噪声水平")
    parser.add_argument("--output-dir", default="results/png_analysis", 
                       help="输出目录")
    parser.add_argument("--format", choices=["png", "pdf"], default="png",
                       help="输出格式 (默认: png)")
    
    args = parser.parse_args()
    
    print(f"🚀 因果追踪分析 - {args.format.upper()} 格式")
    print("=" * 50)
    
    print(f"📥 Loading model: {args.model}")
    
    try:
        mt = EntailmentModelAndTokenizer(args.model)
        print(f"✅ Model loaded successfully!")
        print(f"   Architecture: {mt.model.config.model_type}")
        print(f"   Layers: {mt.num_layers}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    print(f"\n🔍 分析配置:")
    print(f"   Premise: {args.premise}")
    print(f"   Hypothesis: {args.hypothesis}")
    print(f"   数值tokens: {args.numerical_tokens}")
    print(f"   噪声样本: {args.samples}")
    print(f"   噪声水平: {args.noise}")
    print(f"   输出格式: {args.format.upper()}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析列表
    analyses = [
        ("All Components", None, "全组件分析"),
        ("MLP Components", "mlp", "MLP组件分析"), 
        ("Attention Components", "attn", "Attention组件分析")
    ]
    
    print(f"\n🎨 生成 {args.format.upper()} 热力图...")
    
    for analysis_name, kind, description in analyses:
        print(f"   📊 {analysis_name}...")
        
        try:
            if args.format == "png":
                output_file = output_dir / f"causal_trace_{kind or 'all'}.png"
                plot_hidden_flow(
                    mt=mt,
                    premise=args.premise,
                    hypothesis=args.hypothesis,
                    numerical_tokens=args.numerical_tokens,
                    samples=args.samples,
                    noise=args.noise,
                    kind=kind,
                    savepng=str(output_file)
                )
            else:  # PDF
                output_file = output_dir / f"causal_trace_{kind or 'all'}.pdf"
                plot_hidden_flow(
                    mt=mt,
                    premise=args.premise,
                    hypothesis=args.hypothesis,
                    numerical_tokens=args.numerical_tokens,
                    samples=args.samples,
                    noise=args.noise,
                    kind=kind,
                    savepdf=str(output_file)
                )
            
            file_size = output_file.stat().st_size / 1024  # KB
            print(f"   ✅ {description} 保存到: {output_file} ({file_size:.1f} KB)")
            
        except Exception as e:
            print(f"   ❌ {analysis_name} 生成失败: {e}")
            continue
    
    print(f"\n🎉 分析完成!")
    print(f"📁 所有文件保存在: {output_dir}")

if __name__ == "__main__":
    exit(main())