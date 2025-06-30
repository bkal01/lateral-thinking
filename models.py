#!/usr/bin/env python3
"""
Model loading and response generation for evaluation tasks.
"""

import logging
import torch
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name: str, device: str) -> Tuple:
    """
    Load HuggingFace model and tokenizer using Qwen3 pattern.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model {model_name} on device {device}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model using Qwen3 pattern
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        model.eval()
        
        logger.info(f"Successfully loaded model with {model.num_parameters():,} parameters")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 32768) -> Dict[str, str]:
    """
    Generate response from LLM using Qwen3 chat template with thinking mode.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum new tokens to generate
    
    Returns:
        Dictionary with 'thinking_content' and 'content' keys
    """
    try:
        # Prepare the model input using chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking mode for Qwen3
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response with Qwen3 thinking mode parameters
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                do_sample=True
            )
        
        # Extract only the new tokens (remove input)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content (find </think> token - ID 151668)
        try:
            # Find the last occurrence of 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            # No </think> token found, treat all as content
            index = 0
        
        # Decode thinking and content separately
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return {
            'thinking_content': thinking_content,
            'content': content,
            'full_response': tokenizer.decode(output_ids, skip_special_tokens=True)
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error generating response: {e}")
        return {
            'thinking_content': "",
            'content': "",
            'full_response': ""
        }