# method(model, tokenizer, device, input_ids, attention_mask)
def baseline_cot(model, tokenizer, device, input_ids, attention_mask):
    # Generate outputs
    generated_ids = model.generate(
        input_ids=input_ids.squeeze(1).to(device),
        attention_mask=attention_mask.squeeze(1).to(device),
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=2048,
        temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
        top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
        top_k=50,               # Limits vocabulary to top K tokens
        do_sample=True,         # Enable sampling (vs greedy decoding)
        repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def baseline_pot(model, tokenizer, device, input_ids, attention_mask):
    # Generate outputs
    generated_ids = model.generate(
        input_ids=input_ids.squeeze(1).to(device),
        attention_mask=attention_mask.squeeze(1).to(device),
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=2048,
        temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
        top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
        top_k=50,               # Limits vocabulary to top K tokens
        do_sample=True,         # Enable sampling (vs greedy decoding)
        repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)