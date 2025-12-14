from evaluation import PTBTokenizer

# Simulate train.py usage
captions = {
    "1": ["a cat sitting on a mat"],
    "2": ["a dog running in the park"]
}

try:
    print("Testing PTBTokenizer wrapper...")
    tokenized = PTBTokenizer.tokenize(captions)
    print("Tokenization successful!")
    print(tokenized)
    
    # Verify output format
    assert isinstance(tokenized, dict)
    assert "1" in tokenized
    assert isinstance(tokenized["1"], list)
    print("Output format verified.")
    
except Exception as e:
    print("Tokenization failed:", e)
    import traceback
    traceback.print_exc()
