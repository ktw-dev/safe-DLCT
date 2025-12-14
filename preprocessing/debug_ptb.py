from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

tokenizer = PTBTokenizer()
# Try different format: list of dicts with 'caption' and 'image_id'
captions = [
    {'image_id': '1', 'caption': 'a cat sitting on a mat'},
    {'image_id': '2', 'caption': 'a dog running in the park'}
]

try:
    # PTBTokenizer expects a dictionary mapping image_id to list of dicts with 'caption' key?
    # Let's try the format that COCO eval usually expects
    gts = {
        '1': [{'caption': 'a cat sitting on a mat'}],
        '2': [{'caption': 'a dog running in the park'}]
    }
    tokenized = tokenizer.tokenize(gts)
    print("Tokenization successful with gts format!")
    print(tokenized)
except Exception as e:
    print("Tokenization failed with gts format:", e)
