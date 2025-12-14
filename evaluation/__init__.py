__all__ = ['compute_scores', 'PTBTokenizer', 'Cider']

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer as RealPTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import pycocoevalcap.bleu.bleu_scorer as bleu_scorer_module

# Monkey-patch precook to debug "unhashable type: 'list'" error
original_precook = bleu_scorer_module.precook

def debug_precook(s, n=4, out=False):
    try:
        return original_precook(s, n, out)
    except Exception as e:
        print(f"\n[DEBUG] precook failed!")
        print(f"Error: {e}")
        print(f"Input type: {type(s)}")
        print(f"Input value: {s}")
        if isinstance(s, list):
            print("Input is a list! It should be a string.")
        raise e

bleu_scorer_module.precook = debug_precook

class PTBTokenizer:
    @staticmethod
    def tokenize(captions):
        # train.py passes a dict of {image_id: [caption_str, ...]}
        # RealPTBTokenizer expects {image_id: [{'caption': caption_str}, ...]}
        
        # 1. Convert input format
        if isinstance(captions, list):
            # If input is a list, convert to dict with enumerated keys
            captions = {i: c if isinstance(c, list) else [c] for i, c in enumerate(captions)}

        formatted_captions = {}
        for img_id, caps in captions.items():
            clean_caps = []
            for c in caps:
                # Defensive handling for nested lists or non-string types
                if isinstance(c, list):
                    c = " ".join([str(x) for x in c])
                elif not isinstance(c, str):
                    c = str(c)
                clean_caps.append({'caption': c})
            formatted_captions[img_id] = clean_caps
            
        # 2. Instantiate and tokenize
        tokenizer = RealPTBTokenizer()
        tokenized = tokenizer.tokenize(formatted_captions)
        
        # 3. The result is {img_id: [tokenized_cap_str, ...]} which matches what train.py expects
        return tokenized

def compute_scores(gts, gen):
    # Check for empty data
    if not gts or not gen:
        print("Warning: Empty gts or gen. Returning zero scores.")
        return {
            'CIDEr': 0.0,
            'BLEU': [0.0, 0.0, 0.0, 0.0],
            'METEOR': 0.0,
            'ROUGE': 0.0
        }, {}

    metrics = (
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    )
    all_score = {}
    all_scores = {}
    
    for metric, method in metrics:
        try:
            score, scores = metric.compute_score(gts, gen)
            if method == "BLEU_4":
                # BLEU는 1~4 리스트로 반환됨
                all_score["BLEU"] = score
            elif isinstance(method, list):
                # BLEU 1~4 처리
                for m, s in zip(method, score):
                    all_score[m] = s
            else:
                all_score[method] = score
            
            # Fix: Handle list method for all_scores
            if isinstance(method, list):
                for m, s in zip(method, scores):
                    all_scores[m] = s
            else:
                all_scores[method] = scores
                
        except Exception as e:
            print(f"Warning: Metric {method} failed to compute: {e}")
            import traceback
            traceback.print_exc()
            
    # train.py 호환성을 위해 키 이름 매핑
    final_scores = {
        'CIDEr': all_score.get('CIDEr', 0.0),
        'BLEU': all_score.get('BLEU', [0.0, 0.0, 0.0, 0.0]),
        'METEOR': all_score.get('METEOR', 0.0),
        'ROUGE': all_score.get('ROUGE_L', 0.0)
    }
            
    return final_scores, all_scores
