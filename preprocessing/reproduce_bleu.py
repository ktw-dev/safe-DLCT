from evaluation import compute_scores, PTBTokenizer

# Simulate data
gts = {
    '1': ['a cat sitting on a mat'],
    '2': ['a dog running in the park']
}
gen = {
    '1': ['a cat on a mat'],
    '2': ['a dog in the park']
}

print("Tokenizing...")
gts = PTBTokenizer.tokenize(gts)
gen = PTBTokenizer.tokenize(gen)
print("GTS:", gts)
print("GEN:", gen)

if '1' in gts:
    print("Type of gts['1']:", type(gts['1']))
    if len(gts['1']) > 0:
        print("Type of gts['1'][0]:", type(gts['1'][0]))

print("Computing scores...")
try:
    scores, _ = compute_scores(gts, gen)
    print("Scores:", scores)
except Exception as e:
    print("Caught exception:", e)
    import traceback
    traceback.print_exc()

# Test edge case: nested list (if gts_i was list of lists)
print("\nTesting edge case: nested list")
gts_nested = {
    '3': [['nested caption']]
}
try:
    tokenized = PTBTokenizer.tokenize(gts_nested)
    print("Nested tokenized:", tokenized)
    # If tokenized['3'][0] is a list, Bleu will fail
    if isinstance(tokenized['3'][0], list):
        print("WARNING: Tokenizer returned nested list!")
        # Try computing score
        compute_scores(tokenized, tokenized)
except Exception as e:
    print("Caught exception with nested:", e)
