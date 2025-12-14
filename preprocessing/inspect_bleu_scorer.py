import inspect
from pycocoevalcap.bleu.bleu_scorer import BleuScorer

try:
    print(inspect.getsource(BleuScorer))
except Exception as e:
    print("Could not get source:", e)
