"""
Entailment Causal Tracing Experiment
Adapted from ROME's experiments/causal_trace.py for entailment model analysis.

This script implements causal tracing for entailment models to understand
numerical reasoning mechanisms in sentence pairs.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import stats

# Import entailment-specific modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from entailment_rome.entailment_model import EntailmentModelAndTokenizer
from entailment_rome.causal_trace import (
    calculate_hidden_flow,
    trace_with_patch,
    make_inputs,
    predict_from_input
)
from entailment_rome.numerical_utils import (
    find_numerical_range,
    auto_detect_numerical_range,
    get_numerical_token_positions
)
from entailment_util.entailment_globals import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Entailment Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        """Parse noise specification following ROME's pattern"""
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    # Model and data arguments
    aa(
        "--model_name",
        default="microsoft/deberta-v3-large",
        choices=[
            "microsoft/deberta-v3-large",
            "microsoft/deberta-v3-base", 
            "roberta-large-mnli",
            "roberta-base-mnli",
            "bert-base-uncased",
            "textattack/bert-base-uncased-snli"
        ],
        help="Entailment model to analyze"
    )
    aa("--entailment_file", default=None, help="JSON file with entailment cases")
    aa("--output_dir", default="entailment_results/{model_name}/causal_trace")
    
    # Noise and corruption arguments
    aa("--noise_level", default="s3", type=parse_noise_rule, 
       help="Noise level: s3 (spherical gaussian 3x std), m (multivariate), u0.1 (uniform), or float")
    aa("--replace", default=0, type=int, help="Replace tokens with noise (1) vs add noise (0)")
    
    # Analysis arguments
    aa("--samples", default=10, type=int, help="Number of noise samples for causal tracing")
    aa("--max_cases", default=200, type=int, help="Maximum number of cases to process")
    aa("--skip_plots", action="store_true", help="Skip generating individual plots")
    
    # Numerical token arguments
    aa("--numerical_strategy", default="auto", 
       choices=["auto", "premise_only", "hypothesis_only", "both"],
       help="Strategy for selecting numerical tokens to corrupt")
    
    args = parser.parse_args()

    # Setup output directories
    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Load model
    print(f"Loading entailment model: {args.model_name}")
    mt = EntailmentModelAndTokenizer(args.model_name)
    print(f"Model loaded: {mt}")

    # Load entailment cases
    if args.entailment_file is None:
        # Use default entailment cases
        entailment_cases = load_default_entailment_cases()
    else:
        with open(args.entailment_file) as f:
            entailment_cases = json.load(f)

    print(f"Loaded {len(entailment_cases)} entailment cases")

    # Process noise level following ROME's pattern
    noise_level = args.noise_level
    uniform_noise = False
    
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Spherical gaussian noise
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(mt, entailment_cases)
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            # Multivariate gaussian noise
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # t-distribution noise
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
            print(f"Using t-distribution with {degrees} degrees of freedom")
        elif noise_level.startswith("u"):
            # Uniform noise
            uniform_noise = True
            noise_level = float(noise_level[1:])
            print(f"Using uniform noise with magnitude {noise_level}")

    # Process each entailment case
    processed_cases = 0
    for case_idx, case in enumerate(tqdm(entailment_cases)):
        if processed_cases >= args.max_cases:
            break
            
        case_id = case.get("case_id", case_idx)
        premise = case["premise"]
        hypothesis = case["hypothesis"]
        expected_label = case.get("expected_label", "contradiction")
        
        # Process different intervention types
        for kind in [None, "mlp", "attn"]:
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/case_{case_id}{kind_suffix}.npz"
            
            if not os.path.isfile(filename):
                try:
                    result = calculate_hidden_flow(
                        mt,
                        premise,
                        hypothesis,
                        numerical_tokens=case.get("numerical_tokens"),
                        samples=args.samples,
                        kind=kind,
                        noise=noise_level,
                        uniform_noise=uniform_noise,
                        replace=args.replace,
                        expect=expected_label,
                    )
                    
                    # Convert tensors to numpy for saving
                    numpy_result = {
                        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in result.items()
                    }
                    numpy.savez(filename, **numpy_result)
                    
                except Exception as e:
                    print(f"Error processing case {case_id}: {e}")
                    continue
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            
            # Skip cases where model prediction is incorrect
            if not numpy_result["correct_prediction"]:
                tqdm.write(f"Skipping case {case_id}: incorrect prediction")
                continue
            
            # Generate visualization if requested
            if not args.skip_plots:
                plot_result = dict(numpy_result)
                plot_result["kind"] = kind
                pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{case_id}{kind_suffix}.pdf'
                plot_trace_heatmap(plot_result, savepdf=pdfname)
        
        processed_cases += 1

    print(f"Processed {processed_cases} entailment cases")
    print(f"Results saved to: {output_dir}")


def load_default_entailment_cases():
    """
    Load default entailment cases for numerical reasoning analysis.
    These cases focus on numerical differences between premise and hypothesis.
    """
    cases = [
        {
            "case_id": 1,
            "premise": "He was born in 1934.",
            "hypothesis": "He was born in 1935.",
            "expected_label": "contradiction",
            "numerical_tokens": ["1934", "1935"]
        },
        {
            "case_id": 2,
            "premise": "The meeting is at 3:00 PM.",
            "hypothesis": "The meeting is at 4:00 PM.",
            "expected_label": "contradiction",
            "numerical_tokens": ["3:00", "4:00"]
        },
        {
            "case_id": 3,
            "premise": "She scored 85 points in the test.",
            "hypothesis": "She scored 95 points in the test.",
            "expected_label": "contradiction",
            "numerical_tokens": ["85", "95"]
        },
        {
            "case_id": 4,
            "premise": "The temperature is 25 degrees.",
            "hypothesis": "The temperature is 25 degrees.",
            "expected_label": "entailment",
            "numerical_tokens": ["25"]
        },
        {
            "case_id": 5,
            "premise": "There are 100 students in the class.",
            "hypothesis": "There are 200 students in the class.",
            "expected_label": "contradiction",
            "numerical_tokens": ["100", "200"]
        },
        # Add more cases with varying sentence lengths
        {
            "case_id": 6,
            "premise": "The company, which was established in 1995 and has grown significantly over the past decades, reported revenue of $50 million last year.",
            "hypothesis": "The company, which was established in 1996 and has grown significantly over the past decades, reported revenue of $50 million last year.",
            "expected_label": "contradiction",
            "numerical_tokens": ["1995", "1996"]
        },
        {
            "case_id": 7,
            "premise": "In the comprehensive study conducted by researchers at the university, they found that participants who exercised for 30 minutes daily showed improved cognitive performance.",
            "hypothesis": "In the comprehensive study conducted by researchers at the university, they found that participants who exercised for 45 minutes daily showed improved cognitive performance.",
            "expected_label": "contradiction",
            "numerical_tokens": ["30", "45"]
        }
    ]
    
    return cases


def collect_embedding_std(mt, entailment_cases):
    """
    Collect embedding standard deviation for noise calibration.
    Adapted from ROME's collect_embedding_std for entailment models.
    """
    alldata = []
    
    # Sample a subset of cases for efficiency
    sample_cases = entailment_cases[:min(50, len(entailment_cases))]
    
    for case in sample_cases:
        premise = case["premise"]
        hypothesis = case["hypothesis"]
        
        # Create input
        inp = make_inputs(mt.tokenizer, [(premise, hypothesis)])
        
        # Get embeddings using nethook
        from entailment_util import entailment_nethook as nethook
        from entailment_rome.entailment_model import layername
        
        embed_layer = layername(mt.model, 0, "embed")
        
        with nethook.Trace(mt.model, embed_layer) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    
    if alldata:
        alldata = torch.cat(alldata)
        noise_level = alldata.std().item()
    else:
        noise_level = 0.1  # Default fallback
    
    return noise_level


def collect_embedding_gaussian(mt):
    """
    Collect embedding covariance for multivariate gaussian noise.
    Simplified version for entailment models.
    """
    # For now, return a simple noise function
    # This could be expanded to match ROME's full covariance analysis
    return lambda x: 0.1 * torch.randn_like(x)


def collect_embedding_tdist(mt, degrees=3):
    """
    Collect t-distribution noise for embeddings.
    Simplified version for entailment models.
    """
    # For now, return a simple t-distribution approximation
    return lambda x: 0.1 * torch.randn_like(x) * (degrees / (degrees - 2)) ** 0.5


def plot_trace_heatmap(result, savepdf=None, savepng=None, title=None, xlabel=None, modelname=None):
    """
    Plot causal trace heatmap for entailment results.
    Direct port from ROME's plot_trace_heatmap with entailment-specific adaptations.
    
    Args:
        result: Dictionary containing causal tracing results with keys:
            - scores: (tokens, layers) importance matrix
            - low_score: Fully corrupted performance
            - answer: Predicted entailment label
            - kind: Type of intervention (mlp/attn/None)
            - input_tokens: List of token strings
            - numerical_range: Range of numerical tokens to mark
        savepdf: Path to save PDF file (optional)
        savepng: Path to save PNG file (optional)
        title: Custom title for the plot (optional)
        xlabel: Custom x-axis label (optional)
        modelname: Name of the model for labeling (optional)
    """
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result.get("target_label") or result.get("answer")
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    
    # Mark numerical tokens with asterisks (like subject tokens in ROME)
    # Use a set to avoid duplicate marking
    positions_to_mark = set()
    
    # Collect positions from numerical_range
    if "numerical_range" in result:
        positions_to_mark.update(range(*result["numerical_range"]))
    
    # Collect positions from numerical_positions (preferred if available)
    if "numerical_positions" in result:
        positions_to_mark.update(result["numerical_positions"])
    
    # Collect positions from subject_range for backward compatibility
    if "subject_range" in result:
        positions_to_mark.update(range(*result["subject_range"]))
    
    # Apply asterisk marking once per position
    for i in positions_to_mark:
        if i < len(labels):
            labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        # Dynamic figure size based on number of tokens - handle extreme cases
        num_tokens = len(labels)
        num_layers = int(differences.shape[1])
        if num_tokens <= 15:
            # Short sequences: use original scaling
            height = max(2, 2 + (num_tokens - 10) * 0.15)
            width = max(5.0, num_layers * 0.3)
        elif num_tokens <= 30:
            # Medium sequences: more generous scaling
            height = 3 + (num_tokens - 15) * 0.2
            width = max(5.0, num_layers * 0.3)
        elif num_tokens <= 100:
            # Long sequences: aggressive scaling
            height = 6 + (num_tokens - 30) * 0.3
            width = max(5.0, num_layers * 0.3)
        else:
            # Extreme sequences (like 512 tokens): very aggressive scaling
            height = max(25, num_tokens * 0.12)  # At least 0.12 inches per token
            width = max(5.0, num_layers * 0.3)
        # Set reasonable limits but allow for very tall images when needed
        height = min(height, 80)  # Maximum 80 inches tall
        width = max(width, 3.5)   # Minimum 3.5 inches wide
        print(f"   图像尺寸: {width:.1f} x {height:.1f} inches ({num_tokens} tokens)")
        fig, ax = plt.subplots(figsize=(width, height), dpi=200)
        
        # Use ROME's exact color scheme mapping with proper scaling for difference values
        # This shows probability improvement relative to corrupted baseline
        # For difference data: 0 = no improvement, positive = improvement, negative = degradation
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=0,  # 0 represents "no change from corrupted baseline"
        )
        
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        
        # Dynamic font size based on number of tokens - handle extreme cases
        if num_tokens <= 15:
            fontsize = 10
        elif num_tokens <= 30:
            fontsize = 8
        elif num_tokens <= 50:
            fontsize = 6
        elif num_tokens <= 100:
            fontsize = 4
        elif num_tokens <= 200:
            fontsize = 3
        else:
            # For extreme cases like 512 tokens
            fontsize = max(2, 8 - num_tokens // 100)  # Minimum font size of 2
        
        # Set y-axis labels with appropriate font size
        ax.set_yticklabels(labels[:len(differences)], fontsize=fontsize)
        
        # Also adjust x-axis label font size for consistency
        ax.tick_params(axis='x', labelsize=fontsize)
        
        # Entailment-specific model name default
        if not modelname:
            modelname = "Entailment Model"
        
        # Entailment-specific titles and labels
        if not kind:
            ax.set_title("Impact of restoring state after corrupted numerical input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted numerical input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        
        cb = plt.colorbar(h)
        
        # Override with custom title/xlabel if provided
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # Use ROME's exact colorbar labeling approach
            # Note: Using set_title instead of set_xlabel due to matplotlib 3.5.1 bug
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        elif savepng:
            os.makedirs(os.path.dirname(savepng), exist_ok=True)
            plt.savefig(savepng, bbox_inches="tight", dpi=300, format='png')
            plt.close()
        else:
            plt.show()


def plot_hidden_flow(
    mt,
    premise,
    hypothesis,
    numerical_tokens=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
    savepng=None,
    expect=None,
    target_label=None,
    debug_hooks=False,
):
    """
    Plot causal trace for entailment model.
    Direct port from ROME's plot_hidden_flow adapted for entailment.
    
    Args:
        mt: EntailmentModelAndTokenizer instance
        premise: Premise sentence string
        hypothesis: Hypothesis sentence string
        numerical_tokens: List of numerical tokens to focus on (optional)
        samples: Number of noise samples for corruption
        noise: Noise level for corruption
        uniform_noise: Whether to use uniform noise
        window: Window size for interventions
        kind: Type of intervention (mlp/attn/None)
        savepdf: Path to save PDF file (optional)
        savepng: Path to save PNG file (optional)
        expect: Expected entailment label
    """
    result = calculate_hidden_flow(
        mt,
        premise,
        hypothesis,
        numerical_tokens=numerical_tokens,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
        expect=expect,
        target_label=target_label,
        debug_hooks=debug_hooks,
    )
    plot_trace_heatmap(result, savepdf=savepdf, savepng=savepng)


def plot_all_flow(mt, premise, hypothesis, numerical_tokens=None, expect=None):
    """
    Plot causal traces for all intervention types (mlp, attn, full).
    Direct port from ROME's plot_all_flow adapted for entailment.
    
    Args:
        mt: EntailmentModelAndTokenizer instance
        premise: Premise sentence string
        hypothesis: Hypothesis sentence string
        numerical_tokens: List of numerical tokens to focus on (optional)
        expect: Expected entailment label
    """
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, premise, hypothesis, numerical_tokens=numerical_tokens, kind=kind, expect=expect)


def extract_attention_patterns(mt, premise, hypothesis, layer_idx=-1):
    """
    Extract attention patterns from entailment model for numerical token analysis.
    
    Args:
        mt: EntailmentModelAndTokenizer instance
        premise: Premise sentence string
        hypothesis: Hypothesis sentence string
        layer_idx: Layer index to extract attention from (-1 for last layer)
    
    Returns:
        Dictionary containing attention matrices and token information
    """
    # Tokenize input
    inputs = mt.tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(mt.model.device)
    
    # Get attention weights
    with torch.no_grad():
        outputs = mt.model(**inputs, output_attentions=True)
        attentions = outputs.attentions[layer_idx]  # Shape: (batch, heads, seq_len, seq_len)
    
    # Get token information
    tokens = mt.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Find numerical tokens
    numerical_indices = []
    for i, token in enumerate(tokens):
        if re.search(r'\d', token):
            numerical_indices.append(i)
    
    return {
        'attention_matrix': attentions[0].cpu().numpy(),  # (heads, seq_len, seq_len)
        'tokens': tokens,
        'numerical_indices': numerical_indices,
        'input_ids': inputs['input_ids'][0].cpu().numpy(),
        'premise_length': len(mt.tokenizer.encode(premise, add_special_tokens=False)),
        'hypothesis_length': len(mt.tokenizer.encode(hypothesis, add_special_tokens=False))
    }


def plot_attention_rollout(attention_data, savepdf=None, title=None):
    """
    Plot attention rollout visualization focusing on numerical tokens.
    
    Args:
        attention_data: Dictionary from extract_attention_patterns()
        savepdf: Path to save PDF file (optional)
        title: Custom title for the plot (optional)
    """
    attention_matrix = attention_data['attention_matrix']  # (heads, seq_len, seq_len)
    tokens = attention_data['tokens']
    numerical_indices = attention_data['numerical_indices']
    
    # Average attention across heads
    avg_attention = attention_matrix.mean(axis=0)  # (seq_len, seq_len)
    
    # Create attention rollout (cumulative attention flow)
    rollout = numpy.eye(avg_attention.shape[0])
    for i in range(avg_attention.shape[0]):
        rollout = numpy.dot(avg_attention, rollout)
    
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
        
        # Plot 1: Raw attention matrix
        im1 = ax1.imshow(avg_attention, cmap='Blues', aspect='auto')
        ax1.set_title('Average Attention Matrix')
        ax1.set_xlabel('Key Tokens')
        ax1.set_ylabel('Query Tokens')
        
        # Mark numerical tokens
        for idx in numerical_indices:
            ax1.axhline(y=idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Set token labels (every 5th token to avoid crowding)
        step = max(1, len(tokens) // 10)
        tick_indices = list(range(0, len(tokens), step))
        tick_labels = [tokens[i][:8] + ('*' if i in numerical_indices else '') for i in tick_indices]
        
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax1.set_yticks(tick_indices)
        ax1.set_yticklabels(tick_labels)
        
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Plot 2: Attention rollout
        im2 = ax2.imshow(rollout, cmap='Reds', aspect='auto')
        ax2.set_title('Attention Rollout (Cumulative Flow)')
        ax2.set_xlabel('Source Tokens')
        ax2.set_ylabel('Target Tokens')
        
        # Mark numerical tokens
        for idx in numerical_indices:
            ax2.axhline(y=idx, color='blue', linestyle='--', alpha=0.7, linewidth=1)
            ax2.axvline(x=idx, color='blue', linestyle='--', alpha=0.7, linewidth=1)
        
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax2.set_yticks(tick_indices)
        ax2.set_yticklabels(tick_labels)
        
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        if title:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle('Attention Flow Analysis for Numerical Tokens', fontsize=14)
        
        plt.tight_layout()
        
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_premise_hypothesis_attention(attention_data, savepdf=None, title=None):
    """
    Plot attention heatmap showing attention from premise to hypothesis numbers.
    
    Args:
        attention_data: Dictionary from extract_attention_patterns()
        savepdf: Path to save PDF file (optional)
        title: Custom title for the plot (optional)
    """
    attention_matrix = attention_data['attention_matrix']  # (heads, seq_len, seq_len)
    tokens = attention_data['tokens']
    numerical_indices = attention_data['numerical_indices']
    premise_length = attention_data['premise_length']
    
    # Average attention across heads
    avg_attention = attention_matrix.mean(axis=0)
    
    # Find premise and hypothesis numerical tokens
    premise_num_indices = [i for i in numerical_indices if i <= premise_length + 1]  # +1 for [SEP]
    hypothesis_num_indices = [i for i in numerical_indices if i > premise_length + 1]
    
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        
        # Create heatmap
        im = ax.imshow(avg_attention, cmap='YlOrRd', aspect='auto')
        
        # Highlight premise-to-hypothesis numerical attention
        if premise_num_indices and hypothesis_num_indices:
            for p_idx in premise_num_indices:
                for h_idx in hypothesis_num_indices:
                    # Draw rectangle around premise->hypothesis numerical attention
                    rect = plt.Rectangle((p_idx-0.5, h_idx-0.5), 1, 1, 
                                       fill=False, edgecolor='blue', linewidth=2)
                    ax.add_patch(rect)
        
        # Mark all numerical tokens
        for idx in numerical_indices:
            ax.axhline(y=idx, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.axvline(x=idx, color='red', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add separator line between premise and hypothesis
        sep_line = premise_length + 1.5
        ax.axhline(y=sep_line, color='black', linestyle='-', linewidth=2, alpha=0.8)
        ax.axvline(x=sep_line, color='black', linestyle='-', linewidth=2, alpha=0.8)
        
        # Set labels
        ax.set_xlabel('Key Tokens (Source)')
        ax.set_ylabel('Query Tokens (Target)')
        
        # Token labels (show all for shorter sequences, sample for longer ones)
        if len(tokens) <= 20:
            tick_indices = list(range(len(tokens)))
            tick_labels = [tokens[i] + ('*' if i in numerical_indices else '') for i in tick_indices]
        else:
            step = max(1, len(tokens) // 15)
            tick_indices = list(range(0, len(tokens), step))
            tick_labels = [tokens[i][:6] + ('*' if i in numerical_indices else '') for i in tick_indices]
        
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(tick_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Premise-Hypothesis Numerical Attention Flow')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='blue', linewidth=2, label='Premise→Hypothesis Numbers'),
            Patch(facecolor='none', edgecolor='red', linestyle=':', label='Numerical Tokens'),
            Patch(facecolor='none', edgecolor='black', linewidth=2, label='Premise|Hypothesis Boundary')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_length_comparison(short_results, long_results, savepdf=None, title=None):
    """
    Create comparative plots for short vs long sentence processing.
    
    Args:
        short_results: List of causal tracing results for short sentences
        long_results: List of causal tracing results for long sentences
        savepdf: Path to save PDF file (optional)
        title: Custom title for the plot (optional)
    """
    # Calculate average scores for each group
    short_scores = numpy.array([r["scores"] for r in short_results])
    long_scores = numpy.array([r["scores"] for r in long_results])
    
    short_avg = short_scores.mean(axis=0)
    long_avg = long_scores.mean(axis=0)
    
    # Calculate statistical significance using t-test
    from scipy import stats
    
    # Perform t-test for each (token, layer) position
    p_values = numpy.zeros_like(short_avg)
    for i in range(short_avg.shape[0]):
        for j in range(short_avg.shape[1]):
            short_vals = short_scores[:, i, j]
            long_vals = long_scores[:, i, j]
            if len(short_vals) > 1 and len(long_vals) > 1:
                _, p_val = stats.ttest_ind(short_vals, long_vals)
                p_values[i, j] = p_val
            else:
                p_values[i, j] = 1.0  # No significance if insufficient data
    
    # Create significance mask (p < 0.05)
    significance_mask = p_values < 0.05
    
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
        
        # Plot 1: Short sentences average
        im1 = axes[0, 0].imshow(short_avg, cmap='Purples', aspect='auto')
        axes[0, 0].set_title('Short Sentences (≤10 tokens)')
        axes[0, 0].set_xlabel('Layers')
        axes[0, 0].set_ylabel('Tokens')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Plot 2: Long sentences average
        im2 = axes[0, 1].imshow(long_avg, cmap='Purples', aspect='auto')
        axes[0, 1].set_title('Long Sentences (>20 tokens)')
        axes[0, 1].set_xlabel('Layers')
        axes[0, 1].set_ylabel('Tokens')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Plot 3: Difference (Short - Long)
        difference = short_avg - long_avg
        im3 = axes[1, 0].imshow(difference, cmap='RdBu_r', aspect='auto', 
                               vmin=-numpy.abs(difference).max(), vmax=numpy.abs(difference).max())
        axes[1, 0].set_title('Difference (Short - Long)')
        axes[1, 0].set_xlabel('Layers')
        axes[1, 0].set_ylabel('Tokens')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Plot 4: Statistical significance
        im4 = axes[1, 1].imshow(significance_mask.astype(float), cmap='Reds', aspect='auto')
        axes[1, 1].set_title('Statistical Significance (p < 0.05)')
        axes[1, 1].set_xlabel('Layers')
        axes[1, 1].set_ylabel('Tokens')
        
        # Add significance markers
        sig_y, sig_x = numpy.where(significance_mask)
        axes[1, 1].scatter(sig_x, sig_y, c='white', s=10, marker='*', alpha=0.8)
        
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('Short vs Long Sentence Comparison', fontsize=16)
        
        plt.tight_layout()
        
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_attention_comparison_by_length(mt, short_cases, long_cases, layer_idx=-1, savepdf=None):
    """
    Compare attention patterns between short and long sentences with statistical indicators.
    
    Args:
        mt: EntailmentModelAndTokenizer instance
        short_cases: List of (premise, hypothesis) pairs for short sentences
        long_cases: List of (premise, hypothesis) pairs for long sentences
        layer_idx: Layer index to analyze (-1 for last layer)
        savepdf: Path to save PDF file (optional)
    """
    # Extract attention for all cases
    short_attentions = []
    long_attentions = []
    
    for premise, hypothesis in short_cases[:10]:  # Limit to 10 for performance
        att_data = extract_attention_patterns(mt, premise, hypothesis, layer_idx)
        if att_data['numerical_indices']:  # Only include cases with numerical tokens
            short_attentions.append(att_data)
    
    for premise, hypothesis in long_cases[:10]:  # Limit to 10 for performance
        att_data = extract_attention_patterns(mt, premise, hypothesis, layer_idx)
        if att_data['numerical_indices']:  # Only include cases with numerical tokens
            long_attentions.append(att_data)
    
    if not short_attentions or not long_attentions:
        print("Insufficient data with numerical tokens for comparison")
        return
    
    # Calculate average attention to numerical tokens
    def calc_numerical_attention_score(att_data):
        attention_matrix = att_data['attention_matrix'].mean(axis=0)  # Average across heads
        numerical_indices = att_data['numerical_indices']
        
        if not numerical_indices:
            return 0.0
        
        # Calculate average attention TO numerical tokens
        num_attention = attention_matrix[:, numerical_indices].mean()
        return num_attention
    
    short_scores = [calc_numerical_attention_score(att) for att in short_attentions]
    long_scores = [calc_numerical_attention_score(att) for att in long_attentions]
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(short_scores, long_scores)
    
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
        
        # Plot 1: Box plot comparison
        ax1.boxplot([short_scores, long_scores], labels=['Short\n(≤10 tokens)', 'Long\n(>20 tokens)'])
        ax1.set_ylabel('Average Attention to Numerical Tokens')
        ax1.set_title('Attention to Numerical Tokens by Sentence Length')
        
        # Add statistical significance annotation
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        y_max = max(max(short_scores), max(long_scores))
        ax1.text(1.5, y_max * 1.1, f'p = {p_value:.4f} {sig_text}', 
                ha='center', va='bottom', fontsize=12)
        
        # Plot 2: Individual data points
        ax2.scatter([1] * len(short_scores), short_scores, alpha=0.6, label='Short', color='blue')
        ax2.scatter([2] * len(long_scores), long_scores, alpha=0.6, label='Long', color='red')
        ax2.set_xlim(0.5, 2.5)
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['Short\n(≤10 tokens)', 'Long\n(>20 tokens)'])
        ax2.set_ylabel('Average Attention to Numerical Tokens')
        ax2.set_title('Individual Case Comparison')
        ax2.legend()
        
        # Add mean lines
        ax2.axhline(y=numpy.mean(short_scores), xmin=0.1, xmax=0.4, color='blue', linewidth=2, alpha=0.8)
        ax2.axhline(y=numpy.mean(long_scores), xmin=0.6, xmax=0.9, color='red', linewidth=2, alpha=0.8)
        
        plt.tight_layout()
        
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    return {
        'short_scores': short_scores,
        'long_scores': long_scores,
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': (numpy.mean(short_scores) - numpy.mean(long_scores)) / numpy.sqrt(
            (numpy.var(short_scores) + numpy.var(long_scores)) / 2
        )
    }


def create_comprehensive_visualization_report(mt, entailment_cases, output_dir, max_cases=20):
    """
    Generate a comprehensive visualization report including all analysis types.
    
    Args:
        mt: EntailmentModelAndTokenizer instance
        entailment_cases: List of entailment case dictionaries
        output_dir: Directory to save all visualizations
        max_cases: Maximum number of cases to process for performance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Categorize cases by length
    short_cases = []
    long_cases = []
    
    for case in entailment_cases[:max_cases]:
        premise_tokens = len(mt.tokenizer.encode(case["premise"], add_special_tokens=False))
        hypothesis_tokens = len(mt.tokenizer.encode(case["hypothesis"], add_special_tokens=False))
        total_tokens = premise_tokens + hypothesis_tokens
        
        if total_tokens <= 10:
            short_cases.append((case["premise"], case["hypothesis"]))
        elif total_tokens > 20:
            long_cases.append((case["premise"], case["hypothesis"]))
    
    print(f"Found {len(short_cases)} short cases and {len(long_cases)} long cases")
    
    # 1. Generate causal trace heatmaps for sample cases
    print("Generating causal trace heatmaps...")
    for i, case in enumerate(entailment_cases[:5]):  # First 5 cases
        try:
            result = calculate_hidden_flow(
                mt,
                case["premise"],
                case["hypothesis"],
                numerical_tokens=case.get("numerical_tokens"),
                expect=case.get("expected_label"),
            )
            
            plot_trace_heatmap(
                result,
                savepdf=os.path.join(output_dir, f"causal_trace_case_{i+1}.pdf"),
                title=f"Case {i+1}: Causal Trace Analysis"
            )
        except Exception as e:
            print(f"Error processing case {i+1}: {e}")
    
    # 2. Generate attention flow visualizations
    print("Generating attention flow visualizations...")
    for i, case in enumerate(entailment_cases[:3]):  # First 3 cases for attention
        try:
            attention_data = extract_attention_patterns(
                mt, case["premise"], case["hypothesis"]
            )
            
            plot_attention_rollout(
                attention_data,
                savepdf=os.path.join(output_dir, f"attention_rollout_case_{i+1}.pdf"),
                title=f"Case {i+1}: Attention Rollout Analysis"
            )
            
            plot_premise_hypothesis_attention(
                attention_data,
                savepdf=os.path.join(output_dir, f"premise_hypothesis_attention_case_{i+1}.pdf"),
                title=f"Case {i+1}: Premise-Hypothesis Attention Flow"
            )
        except Exception as e:
            print(f"Error processing attention for case {i+1}: {e}")
    
    # 3. Generate length comparison if we have both short and long cases
    if short_cases and long_cases:
        print("Generating length comparison analysis...")
        try:
            # Run causal tracing for short and long cases
            short_results = []
            long_results = []
            
            for premise, hypothesis in short_cases[:5]:
                result = calculate_hidden_flow(mt, premise, hypothesis)
                short_results.append(result)
            
            for premise, hypothesis in long_cases[:5]:
                result = calculate_hidden_flow(mt, premise, hypothesis)
                long_results.append(result)
            
            plot_length_comparison(
                short_results, long_results,
                savepdf=os.path.join(output_dir, "length_comparison_analysis.pdf"),
                title="Short vs Long Sentence Processing Comparison"
            )
            
            # Attention comparison by length
            comparison_stats = plot_attention_comparison_by_length(
                mt, short_cases, long_cases,
                savepdf=os.path.join(output_dir, "attention_length_comparison.pdf")
            )
            
            # Save statistical results
            if comparison_stats:
                with open(os.path.join(output_dir, "length_comparison_stats.json"), 'w') as f:
                    json.dump({
                        'short_mean': float(numpy.mean(comparison_stats['short_scores'])),
                        'long_mean': float(numpy.mean(comparison_stats['long_scores'])),
                        't_statistic': float(comparison_stats['t_statistic']),
                        'p_value': float(comparison_stats['p_value']),
                        'effect_size': float(comparison_stats['effect_size']),
                        'n_short': len(comparison_stats['short_scores']),
                        'n_long': len(comparison_stats['long_scores'])
                    }, indent=2)
        
        except Exception as e:
            print(f"Error in length comparison: {e}")
    
    print(f"Comprehensive visualization report saved to {output_dir}")


def run_batch_analysis(mt, entailment_cases, output_dir, **kwargs):
    """
    Run batch analysis on multiple entailment cases.
    
    Args:
        mt: EntailmentModelAndTokenizer instance
        entailment_cases: List of entailment case dictionaries
        output_dir: Directory to save results
        **kwargs: Additional arguments for calculate_hidden_flow
    """
    results = []
    
    for case_idx, case in enumerate(tqdm(entailment_cases, desc="Processing cases")):
        try:
            result = calculate_hidden_flow(
                mt,
                case["premise"],
                case["hypothesis"],
                numerical_tokens=case.get("numerical_tokens"),
                expect=case.get("expected_label"),
                **kwargs
            )
            
            result["case_id"] = case.get("case_id", case_idx)
            result["premise"] = case["premise"]
            result["hypothesis"] = case["hypothesis"]
            results.append(result)
            
        except Exception as e:
            print(f"Error processing case {case_idx}: {e}")
            continue
    
    # Save batch results
    batch_file = os.path.join(output_dir, "batch_results.json")
    
    # Convert tensors to lists for JSON serialization
    json_results = []
    for result in results:
        json_result = {}
        for k, v in result.items():
            if torch.is_tensor(v):
                json_result[k] = v.detach().cpu().numpy().tolist()
            else:
                json_result[k] = v
        json_results.append(json_result)
    
    with open(batch_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Batch results saved to: {batch_file}")
    return results


if __name__ == "__main__":
    main()