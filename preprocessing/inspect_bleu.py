import inspect
from pycocoevalcap.bleu.bleu import Bleu

try:
    print(inspect.getsource(Bleu.compute_score))
except Exception as e:
    print("Could not get source:", e)
