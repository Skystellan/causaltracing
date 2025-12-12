"""
Numerical token detection utilities for entailment analysis.
Provides functions to identify and locate numerical tokens in sentence pairs.
"""

import re
import torch
from typing import List, Tuple, Optional, Union


def find_numerical_range(tokenizer, token_array, numerical_tokens):
    """
    Find the token range for specified numerical tokens in the input.
    Analogous to ROME's find_token_range but for numerical content.
    
    Args:
        tokenizer: The tokenizer used to encode the input
        token_array: Tensor of token IDs
        numerical_tokens: String or list of numerical tokens to find
    
    Returns:
        Tuple of (start_idx, end_idx) for the numerical token range
        Note: This returns the range from first to last numerical token found
    """
    if isinstance(numerical_tokens, str):
        numerical_tokens = [numerical_tokens]
    
    # Find all positions of numerical tokens
    positions = find_all_numerical_positions(tokenizer, token_array, numerical_tokens)
    
    if not positions:
        return (0, 0)
    
    # Return range from first to last position
    min_pos = min(positions)
    max_pos = max(positions)
    return (min_pos, max_pos + 1)


def find_all_numerical_positions(tokenizer, token_array, numerical_tokens):
    """
    Find all positions of specified numerical tokens in the input.
    
    Args:
        tokenizer: The tokenizer used to encode the input
        token_array: Tensor of token IDs
        numerical_tokens: String or list of numerical tokens to find
    
    Returns:
        List of token positions where numerical tokens are found
    """
    if isinstance(numerical_tokens, str):
        numerical_tokens = [numerical_tokens]
    
    # Decode all tokens to strings
    toks = decode_tokens(tokenizer, token_array)

    # Build a concatenated string and token character offsets.
    # This is robust to models that split numbers into multiple subtokens.
    whole_string = "".join(toks)
    offsets = []
    loc = 0
    for t in toks:
        start = loc
        loc += len(t)
        offsets.append((start, loc))

    def _token_indices_for_char_span(char_start: int, char_end: int):
        idxs = []
        for i, (s, e) in enumerate(offsets):
            if e <= char_start:
                continue
            if s >= char_end:
                break
            idxs.append(i)
        return idxs

    positions = set()
    for num_token in numerical_tokens:
        needle = str(num_token)
        if not needle:
            continue
        search_from = 0
        while True:
            char_loc = whole_string.find(needle, search_from)
            if char_loc == -1:
                break
            span_idxs = _token_indices_for_char_span(char_loc, char_loc + len(needle))
            for i in span_idxs:
                # Skip special tokens
                tok_clean = toks[i].strip()
                if tok_clean == "" or tok_clean in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
                    continue
                positions.add(i)
            search_from = char_loc + 1

    return sorted(positions)


def auto_detect_numerical_range(tokenizer, token_array):
    """
    Automatically detect numerical tokens in the input sequence.
    
    Args:
        tokenizer: The tokenizer used to encode the input
        token_array: Tensor of token IDs
    
    Returns:
        Tuple of (start_idx, end_idx) for the first numerical token range found
    """
    toks = decode_tokens(tokenizer, token_array)
    
    # Look for numerical patterns in tokens
    for i, token in enumerate(toks):
        if is_numerical_token(token):
            # Found a numerical token, determine its range
            start_idx = i
            end_idx = i + 1
            
            # Check if subsequent tokens are also part of the number
            while end_idx < len(toks) and is_numerical_continuation(toks[end_idx]):
                end_idx += 1
            
            return (start_idx, end_idx)
    
    # If no numerical tokens found, return empty range
    return (0, 0)


def find_all_numerical_ranges(tokenizer, token_array):
    """
    Find all numerical token ranges in the input sequence.
    
    Args:
        tokenizer: The tokenizer used to encode the input
        token_array: Tensor of token IDs
    
    Returns:
        List of tuples [(start_idx, end_idx), ...] for all numerical ranges
    """
    toks = decode_tokens(tokenizer, token_array)
    ranges = []
    i = 0
    
    while i < len(toks):
        if is_numerical_token(toks[i]):
            start_idx = i
            end_idx = i + 1
            
            # Extend range for multi-token numbers
            while end_idx < len(toks) and is_numerical_continuation(toks[end_idx]):
                end_idx += 1
            
            ranges.append((start_idx, end_idx))
            i = end_idx
        else:
            i += 1
    
    return ranges


def is_numerical_token(token):
    """
    Check if a token represents numerical content.
    
    Args:
        token: String token to check
    
    Returns:
        Boolean indicating if token is numerical
    """
    # Remove common tokenizer artifacts
    clean_token = token.strip().replace("Ġ", "").replace("▁", "")
    
    if not clean_token:
        return False
    
    # Check for various numerical patterns
    patterns = [
        r'^\d+$',                    # Pure integers: 123, 45
        r'^\d+\.\d+$',              # Decimals: 12.34, 0.5
        r'^\d+,\d+$',               # Comma-separated: 1,234
        r'^\d+\.\d+\.\d+$',         # Dates: 12.3.2021
        r'^\d+/\d+/\d+$',           # Dates: 12/3/2021
        r'^\d+-\d+-\d+$',           # Dates: 2021-03-12
        r'^\d+:\d+$',               # Times: 12:34
        r'^\d+%$',                  # Percentages: 25%
        r'^\$\d+$',                 # Currency: $100
        r'^\d+\w{2}$',              # Ordinals: 1st, 2nd, 3rd
        r'^-?\d+$',                 # Negative numbers: -123
        r'^-?\d+\.\d+$',            # Negative decimals: -12.34
    ]
    
    for pattern in patterns:
        if re.match(pattern, clean_token):
            return True
    
    # Check for written numbers
    written_numbers = {
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 
        'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
        'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
        'billion', 'trillion', 'first', 'second', 'third', 'fourth', 'fifth'
    }
    
    if clean_token.lower() in written_numbers:
        return True
    
    return False


def is_numerical_continuation(token):
    """
    Check if a token is a continuation of a numerical expression.
    
    Args:
        token: String token to check
    
    Returns:
        Boolean indicating if token continues a number
    """
    clean_token = token.strip().replace("Ġ", "").replace("▁", "")
    
    # Common numerical continuations
    continuations = [
        r'^\d+$',           # More digits
        r'^,\d+$',          # Comma continuation: ,000
        r'^\.\d+$',         # Decimal continuation: .50
        r'^:\d+$',          # Time continuation: :30
        r'^/\d+$',          # Fraction/date continuation: /2021
        r'^-\d+$',          # Range continuation: -2021
        r'^%$',             # Percentage symbol
        r'^st$|^nd$|^rd$|^th$',  # Ordinal suffixes
    ]
    
    for pattern in continuations:
        if re.match(pattern, clean_token):
            return True
    
    return False


def find_numerical_tokens_in_premise_hypothesis(tokenizer, premise, hypothesis):
    """
    Find numerical tokens separately in premise and hypothesis.
    
    Args:
        tokenizer: The tokenizer to use
        premise: Premise sentence string
        hypothesis: Hypothesis sentence string
    
    Returns:
        Dictionary with premise_ranges and hypothesis_ranges
    """
    # Tokenize premise and hypothesis separately
    premise_tokens = tokenizer.encode(premise, add_special_tokens=False)
    hypothesis_tokens = tokenizer.encode(hypothesis, add_special_tokens=False)
    
    # Find numerical ranges in each
    premise_ranges = find_all_numerical_ranges(tokenizer, torch.tensor(premise_tokens))
    hypothesis_ranges = find_all_numerical_ranges(tokenizer, torch.tensor(hypothesis_tokens))
    
    # Adjust hypothesis ranges to account for premise + special tokens
    # This depends on how the tokenizer combines premise and hypothesis
    premise_length = len(premise_tokens)
    
    # Account for [CLS] token at the beginning (typically index 0)
    # and [SEP] token between premise and hypothesis
    sep_tokens = 2  # [CLS] + [SEP]
    adjusted_hypothesis_ranges = [
        (start + premise_length + sep_tokens, end + premise_length + sep_tokens)
        for start, end in hypothesis_ranges
    ]
    
    return {
        'premise_ranges': premise_ranges,
        'hypothesis_ranges': adjusted_hypothesis_ranges,
        'all_ranges': premise_ranges + adjusted_hypothesis_ranges
    }


def extract_numerical_values(tokenizer, token_array, numerical_range):
    """
    Extract the actual numerical values from a token range.
    
    Args:
        tokenizer: The tokenizer used to encode the input
        token_array: Tensor of token IDs
        numerical_range: Tuple of (start_idx, end_idx)
    
    Returns:
        String representation of the numerical value
    """
    start_idx, end_idx = numerical_range
    if start_idx >= end_idx or end_idx > len(token_array):
        return ""
    
    # Extract tokens in the range
    range_tokens = token_array[start_idx:end_idx]
    
    # Decode to string
    numerical_string = tokenizer.decode(range_tokens, skip_special_tokens=True)
    
    return numerical_string.strip()


def decode_tokens(tokenizer, token_array):
    """
    Decode token arrays back to strings.
    Helper function adapted from ROME's decode_tokens.
    """
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def create_entailment_inputs(tokenizer, premise, hypothesis, device="auto"):
    """
    Create tokenized inputs for entailment models from premise-hypothesis pairs.
    
    Args:
        tokenizer: The tokenizer for the entailment model
        premise: Premise sentence string
        hypothesis: Hypothesis sentence string
        device: Device to place tensors on (default: "cuda")
    
    Returns:
        Dictionary with input_ids, attention_mask, and token_type_ids
    """
    # Handle device availability
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Tokenize premise-hypothesis pair
    encoded = tokenizer(
        premise,
        hypothesis,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
        add_special_tokens=True
    )
    
    # Move to device
    return {k: v.to(device) for k, v in encoded.items()}


def get_numerical_token_positions(tokenizer, premise, hypothesis, target_numbers=None):
    """
    Get positions of specific numerical tokens in premise-hypothesis pairs.
    
    Args:
        tokenizer: The tokenizer to use
        premise: Premise sentence string
        hypothesis: Hypothesis sentence string  
        target_numbers: List of specific numbers to find (optional)
    
    Returns:
        Dictionary with detailed position information
    """
    # Create full input as the model would see it
    encoded = tokenizer(premise, hypothesis, return_tensors="pt", add_special_tokens=True)
    token_ids = encoded["input_ids"][0]
    
    # Get all numerical ranges
    all_ranges = find_all_numerical_ranges(tokenizer, token_ids)
    
    result = {
        'token_ids': token_ids,
        'all_numerical_ranges': all_ranges,
        'numerical_values': []
    }
    
    # Extract actual values for each range
    for range_tuple in all_ranges:
        value = extract_numerical_values(tokenizer, token_ids, range_tuple)
        result['numerical_values'].append({
            'range': range_tuple,
            'value': value
        })
    
    # If specific target numbers are provided, find their positions
    if target_numbers:
        result['target_positions'] = {}
        for target in target_numbers:
            target_range = find_numerical_range(tokenizer, token_ids, str(target))
            if target_range != (0, 0):
                result['target_positions'][str(target)] = target_range
    
    return result