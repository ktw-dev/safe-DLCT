from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

tokenizer = PTBTokenizer()
captions = {
    "1": ["a cat sitting on a mat"],
    "2": ["a dog running in the park"]
}

try:
    tokenized = tokenizer.tokenize(captions)
    print("Tokenization successful!")
    print(tokenized)
except Exception as e:
    print("Tokenization failed:", e)
