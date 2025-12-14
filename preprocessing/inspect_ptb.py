from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import inspect

print("PTBTokenizer type:", type(PTBTokenizer))
print("PTBTokenizer.tokenize type:", type(PTBTokenizer.tokenize))
try:
    print("Signature:", inspect.signature(PTBTokenizer.tokenize))
except Exception as e:
    print("Could not get signature:", e)

print("Is callable?", callable(PTBTokenizer.tokenize))
