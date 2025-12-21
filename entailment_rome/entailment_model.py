"""
EntailmentModelAndTokenizer class for entailment model analysis.
Adapted from ROME's ModelAndTokenizer for sequence classification models.
"""

import re
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DebertaV2ForSequenceClassification,
    RobertaForSequenceClassification, 
    BertForSequenceClassification
)

# Try to import nethook, but provide fallback for testing
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from entailment_util import entailment_nethook as nethook
except ImportError:
    # Fallback for testing - create a mock nethook
    class MockNethook:
        @staticmethod
        def set_requires_grad(requires_grad, model):
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    param.requires_grad = requires_grad
    nethook = MockNethook()


class EntailmentModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    an entailment model and tokenizer. Counts the number of layers
    and provides layer name utilities for sequence classification models.
    
    Adapted from ROME's ModelAndTokenizer for entailment tasks.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model is None:
            assert model_name is not None
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                low_cpu_mem_usage=low_cpu_mem_usage, 
                torch_dtype=torch_dtype
            )
            nethook.set_requires_grad(False, model)
            
            # Move to appropriate device
            if torch.cuda.is_available():
                model.eval().cuda()
            else:
                model.eval().cpu()
        
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name or "unknown"
        
        # Detect architecture and set layer names accordingly
        self.architecture = self._detect_architecture()
        self.layer_names = self._get_layer_names()
        self.num_layers = len(self.layer_names)

    def _detect_architecture(self):
        """Detect the model architecture (DeBERTa, RoBERTa, BERT, etc.)"""
        if hasattr(self.model, "deberta"):
            return "deberta"
        elif hasattr(self.model, "roberta"):
            return "roberta"
        elif hasattr(self.model, "bert"):
            return "bert"
        else:
            # Try to infer from model class name
            model_class = type(self.model).__name__.lower()
            if "deberta" in model_class:
                return "deberta"
            elif "roberta" in model_class:
                return "roberta"
            elif "bert" in model_class:
                return "bert"
            else:
                raise ValueError(f"Unsupported model architecture: {type(self.model)}")

    def _get_layer_names(self):
        """Get transformer layer names based on architecture"""
        layer_names = []
        
        if self.architecture == "deberta":
            # DeBERTa structure: deberta.encoder.layer.{num}
            for n, m in self.model.named_modules():
                if re.match(r"^deberta\.encoder\.layer\.\d+$", n):
                    layer_names.append(n)
        elif self.architecture == "roberta":
            # RoBERTa structure: roberta.encoder.layer.{num}
            for n, m in self.model.named_modules():
                if re.match(r"^roberta\.encoder\.layer\.\d+$", n):
                    layer_names.append(n)
        elif self.architecture == "bert":
            # BERT structure: bert.encoder.layer.{num}
            for n, m in self.model.named_modules():
                if re.match(r"^bert\.encoder\.layer\.\d+$", n):
                    layer_names.append(n)
        
        # Sort layer names by layer number
        layer_names.sort(key=lambda x: int(x.split('.')[-1]))
        return layer_names

    def __repr__(self):
        return (
            f"EntailmentModelAndTokenizer("
            f"model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"architecture: {self.architecture}, "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    """
    Generate layer names for different entailment model architectures.
    Adapted from ROME's layername() function for sequence classification models.
    
    Args:
        model: The entailment model
        num: Layer number (0-indexed)
        kind: Type of layer component ('embed', 'attn', 'mlp', None for full layer)
    
    Returns:
        String name of the layer component
    """
    # Detect architecture
    if hasattr(model, "deberta"):
        if kind == "embed":
            return "deberta.embeddings.word_embeddings"
        elif kind == "attn":
            return f"deberta.encoder.layer.{num}.attention.output.dense"
        elif kind == "mlp":
            return f"deberta.encoder.layer.{num}.output.dense"
        else:
            # Full layer
            return f"deberta.encoder.layer.{num}"
    
    elif hasattr(model, "roberta"):
        if kind == "embed":
            return "roberta.embeddings.word_embeddings"
        elif kind == "attn":
            return f"roberta.encoder.layer.{num}.attention.output.dense"
        elif kind == "mlp":
            return f"roberta.encoder.layer.{num}.output.dense"
        else:
            # Full layer
            return f"roberta.encoder.layer.{num}"
    
    elif hasattr(model, "bert"):
        if kind == "embed":
            return "bert.embeddings.word_embeddings"
        elif kind == "attn":
            return f"bert.encoder.layer.{num}.attention.output.dense"
        elif kind == "mlp":
            return f"bert.encoder.layer.{num}.output.dense"
        else:
            # Full layer
            return f"bert.encoder.layer.{num}"
    
    else:
        # Try to infer from model class name
        model_class = type(model).__name__.lower()
        if "deberta" in model_class:
            if kind == "embed":
                return "deberta.embeddings.word_embeddings"
            elif kind == "attn":
                return f"deberta.encoder.layer.{num}.attention.output.dense"
            elif kind == "mlp":
                return f"deberta.encoder.layer.{num}.output.dense"
            else:
                return f"deberta.encoder.layer.{num}"
        elif "roberta" in model_class:
            if kind == "embed":
                return "roberta.embeddings.word_embeddings"
            elif kind == "attn":
                return f"roberta.encoder.layer.{num}.attention.output.dense"
            elif kind == "mlp":
                return f"roberta.encoder.layer.{num}.output.dense"
            else:
                return f"roberta.encoder.layer.{num}"
        elif "bert" in model_class:
            if kind == "embed":
                return "bert.embeddings.word_embeddings"
            elif kind == "attn":
                return f"bert.encoder.layer.{num}.attention.output.dense"
            elif kind == "mlp":
                return f"bert.encoder.layer.{num}.output.dense"
            else:
                return f"bert.encoder.layer.{num}"
        else:
            raise ValueError(f"Unknown transformer structure: {type(model)}")


def get_embedding_layer_name(model):
    """Get the name of the embedding layer for the given model architecture"""
    return layername(model, 0, kind="embed")


def get_classifier_layer_name(model):
    """Get the name of the classification head for the given model architecture"""
    if hasattr(model, "deberta"):
        return "classifier"
    elif hasattr(model, "roberta"):
        return "classifier"
    elif hasattr(model, "bert"):
        return "classifier"
    else:
        # Try to infer from model class name
        model_class = type(model).__name__.lower()
        if any(arch in model_class for arch in ["deberta", "roberta", "bert"]):
            return "classifier"
        else:
            raise ValueError(f"Unknown classifier structure: {type(model)}")