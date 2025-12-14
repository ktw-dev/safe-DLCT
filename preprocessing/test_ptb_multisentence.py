from evaluation import PTBTokenizer

# Test case: Caption with multiple sentences
captions = {
    "1": ["A dog. A cat."],
    "2": ["Simple caption"]
}

print("Testing PTBTokenizer with multi-sentence caption...")
try:
    tokenized = PTBTokenizer.tokenize(captions)
    print("Tokenized output:", tokenized)
    
    # Check types
    for img_id, caps in tokenized.items():
        for i, cap in enumerate(caps):
            print(f"ID {img_id} Cap {i} Type: {type(cap)}")
            if isinstance(cap, list):
                print(f"  -> WARNING: Caption is a list! {cap}")
            else:
                print(f"  -> Content: {cap}")

except Exception as e:
    print("Tokenization failed:", e)
