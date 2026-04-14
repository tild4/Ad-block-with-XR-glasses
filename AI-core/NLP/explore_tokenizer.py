"""Loads the KB-Lab Swedish ELCTRA tokenizer and inspects what it does to sampel sentences.
Run from repo root with venv activated:
ex: python3 Ad-block-with-XR-glasses/AI-core/NLP/explore_tokenizer.py
"""

from transformers import AutoTokenizer

MODEL_NAME = "KBLab/electra-small-swedish-cased-discriminator"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Vocabulary size: {tokenizer.vocab_size}")
print("Special tokens:", tokenizer.special_tokens_map)


print("Tokenizing a sample sentence")
sample_text = "Halva priset på laptops denna vecka"                                                                                                          
print(f"Input text: '{sample_text}'")

tokens = tokenizer.tokenize(sample_text)                                                                                                                   
print(f"Tokens:     {tokens}")
                                                                                                                                                            
encoded = tokenizer(sample_text)
print(f"Input IDs:       {encoded['input_ids']}")                                                                                                            
print(f"Attention mask:  {encoded['attention_mask']}") 



encoded_padded = tokenizer(                                                                                                                                  
    sample_text,
    padding="max_length",                                                                                                                                    
    max_length=16,                                                                                                                                         
    truncation=True,                                                                                                                                         
)
print(f"Padded input IDs:      {encoded_padded['input_ids']}")                                                                                               
print(f"Padded attention mask: {encoded_padded['attention_mask']}")                                                                                          
print(f"Decoded back:          '{tokenizer.decode(encoded_padded['input_ids'])}'")

print("-----------")

noisy_samples = [                                                                                                                                            
    "B0ka b0rd nu h0s Café S01",                                                                                                                           
    "DITt K0nt0 spÄrrAs iN0m 24 timmR",
    "Halva priset på pizzor",                                                                                                                                
]                                                                                                                                                            
for text in noisy_samples:                                                                                                                                   
    tokens = tokenizer.tokenize(text)                                                                                                                        
    print(f"Input:  '{text}'")                                                                                                                             
    print(f"Tokens: {tokens}")                                                                                                                               
    print()
