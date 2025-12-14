import inspect
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

try:
    print(inspect.getsource(PTBTokenizer.tokenize))
except Exception as e:
    print("Could not get source:", e)
