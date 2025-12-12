"""
Causal tracing functions for entailment models.
Adapted from ROME's causal tracing methodology for sequence classification.
"""

import numpy
import torch
from collections import defaultdict
from .entailment_model import layername

# Try to import nethook, but provide fallback for testing
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from entailment_util import entailment_nethook as nethook
except ImportError:
    # Fallback for testing - create a mock nethook
    class MockNethook:
        class TraceDict:
            def __init__(self, model, layers, edit_output=None):
                self.model = model
                self.layers = layers
                self.edit_output = edit_output
                
            def __enter__(self):
                return self
                
            def __exit__(self, *args):
                pass
                
            def __getitem__(self, key):
                # Mock trace output
                class MockTrace:
                    def __init__(self):
                        self.output = torch.zeros(1, 10, 768)  # Mock tensor
                return MockTrace()
        
        @staticmethod
        def set_requires_grad(requires_grad, model):
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    param.requires_grad = requires_grad
    
    nethook = MockNethook()


def trace_with_patch(
    model,  # The entailment model
    inp,  # A set of tokenized sentence pair inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Target entailment class indices to collect probabilities for
    tokens_to_mix,  # Range of tokens to corrupt (begin, end) - typically numerical tokens
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
    debug_hooks=False,
):
    """
    Runs a single causal trace for entailment models. Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs. The argument tokens_to_mix specifies tokens to
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run. This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.

    Adapted from ROME's trace_with_patch() for entailment prediction instead of next-token prediction.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    _printed = set()

    def _shape_str(val):
        try:
            h = untuple(val)
            if isinstance(h, torch.Tensor):
                return str(tuple(h.shape))
        except Exception:
            pass
        return str(type(val))

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            if debug_hooks and layer not in _printed:
                _printed.add(layer)
                b, e = tokens_to_mix if tokens_to_mix is not None else (None, None)
                print(f"[debug-hooks] layer={layer} output_shape={_shape_str(x)} corrupt_tokens={b}:{e}")
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        if debug_hooks and layer not in _printed:
            _printed.add(layer)
            print(f"[debug-hooks] layer={layer} output_shape={_shape_str(x)} patch_tokens={patch_spec[layer]}")
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # For entailment models, we report softmax probabilities for the target classes
    # Instead of next-token logits, we use the classification logits
    probs = torch.softmax(outputs_exp.logits[1:], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def make_inputs(tokenizer, premise_hypothesis_pairs, device="auto"):
    """
    Create tokenized inputs for entailment models from premise-hypothesis pairs.
    
    Args:
        tokenizer: The tokenizer for the entailment model
        premise_hypothesis_pairs: List of (premise, hypothesis) tuples or 
                                 list of strings in format "premise [SEP] hypothesis"
        device: Device to place tensors on
    
    Returns:
        Dictionary with input_ids, attention_mask, and token_type_ids
    """
    if isinstance(premise_hypothesis_pairs[0], tuple):
        # List of (premise, hypothesis) tuples
        premises = [pair[0] for pair in premise_hypothesis_pairs]
        hypotheses = [pair[1] for pair in premise_hypothesis_pairs]
        
        # Tokenize premise-hypothesis pairs
        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
    else:
        # List of strings - assume they're already formatted or single sentences
        encoded = tokenizer(
            premise_hypothesis_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
    
    # Handle device availability
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Move to device
    return {k: v.to(device) for k, v in encoded.items()}


def decode_tokens(tokenizer, token_array):
    """
    Decode token arrays back to strings.
    Adapted from ROME's decode_tokens function.
    """
    # Use the implementation from numerical_utils to avoid duplication
    return numerical_decode_tokens(tokenizer, token_array)


def predict_from_input(model, inp):
    """
    Get predictions from entailment model input.
    Adapted from ROME's predict_from_input for classification instead of generation.
    
    Returns:
        preds: Predicted class indices
        p: Prediction probabilities for the predicted classes
    """
    out = model(**inp)
    probs = torch.softmax(out.logits, dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def trace_important_states(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
    debug_hooks=False,
):
    """
    Trace the importance of individual states across all layers and tokens.
    Adapted from ROME's trace_important_states for entailment models.
    """
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                debug_hooks=debug_hooks,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
    debug_hooks=False,
):
    """
    Trace the importance of windowed states across layers.
    Adapted from ROME's trace_important_window for entailment models.
    """
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                debug_hooks=debug_hooks,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def calculate_hidden_flow(
    mt,
    premise,
    hypothesis,
    numerical_tokens=None,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
    target_label=None,
    debug_hooks=False,
):
    """
    Runs causal tracing over every token/layer combination in the entailment network
    and returns a dictionary numerically summarizing the results.
    
    Adapted from ROME's calculate_hidden_flow for entailment models with numerical reasoning.
    
    Args:
        mt: EntailmentModelAndTokenizer instance
        premise: The premise sentence
        hypothesis: The hypothesis sentence  
        numerical_tokens: List of numerical tokens to focus on (if None, auto-detect)
        samples: Number of noise samples to use
        noise: Noise level for corruption
        token_range: Specific token range to analyze
        uniform_noise: Whether to use uniform noise
        replace: Whether to replace tokens with noise vs add noise
        window: Window size for windowed analysis
        kind: Type of component to analyze ('mlp', 'attn', None for full layer)
        expect: Expected entailment label for validation
    
    Returns:
        Dictionary with causal tracing results including:
        - scores: (tokens, layers) importance matrix
        - low_score: Performance with full corruption
        - high_score: Performance without corruption
        - input_ids: Token IDs
        - input_tokens: Decoded tokens
        - numerical_range: Range of numerical tokens
        - answer: Predicted entailment label
        - correct_prediction: Whether model prediction matches expectation
    """
    # Create inputs - replicate the premise-hypothesis pair for batch processing
    inp = make_inputs(mt.tokenizer, [(premise, hypothesis)] * (samples + 1))
    
    # Prefer the model's own label mapping when available.
    # Different MNLI checkpoints may permute label ids.
    def _normalize_label_name(name: str) -> str:
        return str(name).strip().lower()

    config_id2label = getattr(getattr(mt, "model", None), "config", None)
    config_id2label = getattr(config_id2label, "id2label", None)
    config_label2id = getattr(getattr(mt, "model", None), "config", None)
    config_label2id = getattr(config_label2id, "label2id", None)

    label_map = None
    label_to_id = None
    if isinstance(config_id2label, dict) and len(config_id2label) >= 3:
        try:
            label_map = {int(k): _normalize_label_name(v) for k, v in config_id2label.items()}
        except Exception:
            label_map = None
    if isinstance(config_label2id, dict) and len(config_label2id) >= 3:
        try:
            label_to_id = {_normalize_label_name(k): int(v) for k, v in config_label2id.items()}
        except Exception:
            label_to_id = None

    # Fallback mapping (common, but not guaranteed)
    if label_map is None or label_to_id is None:
        label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
        label_to_id = {v: k for k, v in label_map.items()}

    with torch.no_grad():
        pred_label_id, base_score = [d[0] for d in predict_from_input(mt.model, inp)]

    # Select which class probability to track in causal tracing.
    # Default to contradiction to match typical numerical inconsistency probes.
    if target_label is None:
        target_label = "contradiction"
    target_label = _normalize_label_name(target_label)
    if target_label not in label_to_id:
        raise ValueError(
            f"Unknown target_label: {target_label}. Expected one of {sorted(label_to_id.keys())}"
        )
    target_label_id = torch.tensor(label_to_id[target_label], device=pred_label_id.device)

    # Decode the predicted answer (for reporting only)
    answer = label_map.get(pred_label_id.item(), f"class_{pred_label_id.item()}")
    
    # Check if prediction matches expectation
    if expect is not None and answer.strip() != expect:
        return dict(correct_prediction=False)
    
    # Find numerical token range and positions
    if numerical_tokens is not None:
        e_range = find_numerical_range(mt.tokenizer, inp["input_ids"][0], numerical_tokens)
        # Also get individual positions for precise marking
        from .numerical_utils import find_all_numerical_positions
        numerical_positions = find_all_numerical_positions(mt.tokenizer, inp["input_ids"][0], numerical_tokens)
    else:
        # Auto-detect numerical tokens
        e_range = auto_detect_numerical_range(mt.tokenizer, inp["input_ids"][0])
        numerical_positions = []
    
    # Handle token_range specification
    if token_range == "numerical_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    
    # Measure performance with full corruption (no restoration)
    low_score = trace_with_patch(
        mt.model,
        inp,
        [],
        target_label_id,
        e_range,
        noise=noise,
        uniform_noise=uniform_noise,
        debug_hooks=debug_hooks,
    ).item()
    
    # Perform systematic causal tracing
    if not kind:
        restored_scores = trace_important_states(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            target_label_id,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
            debug_hooks=debug_hooks,
        )
    else:
        restored_scores = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            target_label_id,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
            debug_hooks=debug_hooks,
        )
    
    # Calculate the actual differences: restored_score - corrupted_baseline
    # This shows how much each restoration improves over the corrupted baseline
    differences = restored_scores.detach().cpu() - low_score
    
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        numerical_range=e_range,
        numerical_positions=numerical_positions,
        answer=answer,
        target_label=target_label,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )


# Import numerical detection utilities
try:
    from .numerical_utils_fixed import (
        find_numerical_range,
        auto_detect_numerical_range,
        decode_tokens as numerical_decode_tokens
    )
except ImportError:
    from .numerical_utils import (
        find_numerical_range,
        auto_detect_numerical_range,
        decode_tokens as numerical_decode_tokens
    )