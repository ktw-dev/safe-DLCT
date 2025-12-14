from pycocoevalcap.bleu.bleu_scorer import precook, cook_refs

print("Testing precook with string...")
try:
    precook("this is a test")
    print("Success")
except Exception as e:
    print("Failed:", e)

print("\nTesting precook with list (should raise AttributeError)...")
try:
    precook(["this", "is", "list"])
except Exception as e:
    print(f"Caught expected {type(e).__name__}: {e}")

print("\nTesting cook_refs with list of strings...")
try:
    cook_refs(["this is a test", "another ref"])
    print("Success")
except Exception as e:
    print("Failed:", e)

print("\nTesting cook_refs with list of lists (should raise AttributeError)...")
try:
    cook_refs([["this", "is", "list"]])
except Exception as e:
    print(f"Caught expected {type(e).__name__}: {e}")

print("\nTesting cook_refs with mixed types...")
try:
    cook_refs(["valid string", ["invalid", "list"]])
except Exception as e:
    print(f"Caught expected {type(e).__name__}: {e}")
