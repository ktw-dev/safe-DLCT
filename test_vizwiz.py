import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
import torch
import argparse, os, pickle
import numpy as np
from tqdm import tqdm
import time
from data import build_image_field
from models import model_factory

def evaluate_metrics(model, dataloader, text_field, device):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, ((images, boxes, grids, masks, classes, depth_regions, depth_grids), caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            boxes = boxes.to(device)
            grids = grids.to(device)
            masks = masks.to(device)
            classes = classes.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1,
                                           **{'boxes': boxes, 'grids': grids, 'masks': masks, 'classes': classes})
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            
            if it == 0:
                print("\n--- Debug: Generated vs Ground Truth (First Batch) ---")
                for i in range(min(5, len(caps_gen))):
                    print(f"Gen: {gen['0_%d' % i][0]}")
                    print(f"GT:  {gts['0_%d' % i]}")
                    print("-" * 30)
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Safe-DLCT VizWiz Evaluation')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--features_path', type=str, required=True, help='Path to features directory')
    parser.add_argument('--annotation_folder', type=str, required=True, help='Path to annotation directory')
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--grid_on', action='store_true', default=False)
    parser.add_argument('--max_detections', type=int, default=50)
    parser.add_argument('--dim_feats', type=int, default=2048)
    parser.add_argument('--image_field', type=str, default="ImageAllFieldWithMask")
    parser.add_argument('--model', type=str, default="DLCT")
    parser.add_argument('--grid_embed', action='store_true', default=True)
    parser.add_argument('--box_embed', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to specific checkpoint to load')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Split to evaluate on')
    
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Safe-DLCT VizWiz Evaluation')

    # Pipeline for image regions
    image_field = build_image_field(args)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    from data import VizWiz
    ann_file = os.path.join(args.annotation_folder, 'dataset_vizwiz.json')
    img_root = 'data/vizwiz' # This might need adjustment or be passed as arg if needed, but usually fixed structure
    dataset = VizWiz(image_field, text_field, img_root, ann_file)
        
    train_dataset, val_dataset, test_dataset = dataset.splits

    if args.split == 'train':
        target_dataset = train_dataset
    elif args.split == 'val':
        target_dataset = val_dataset
    else:
        target_dataset = test_dataset

    if len(target_dataset) == 0:
        print(f"Error: The selected split '{args.split}' is empty or has no samples with captions.")
        print("Note: The official VizWiz test set does not have public annotations.")
        print("To evaluate metrics, please use '--split val' or provide a dataset with test annotations.")
        exit(1)

    vocab_path = 'vocab_vizwiz.pkl'
    if os.path.isfile(vocab_path):
            print(f"Loading vocabulary from {vocab_path}")
            text_field.vocab = pickle.load(open(vocab_path, 'rb'))
    else:
            print("Building vocabulary for VizWiz")
            text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
            pickle.dump(text_field.vocab, open(vocab_path, 'wb'))

    # Model and dataloaders
    Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention = model_factory(args)
    encoder = TransformerEncoder(args.n_layer, 0, attention_module=ScaledDotProductAttention,
                                 d_in=args.dim_feats,
                                 d_in_region=1024, 
                                 d_k=args.d_k,
                                 d_v=args.d_v,
                                 h=args.head,
                                 d_model=args.d_model
                                 )
    decoder = TransformerDecoderLayer(len(text_field.vocab), 200, args.n_layer, text_field.vocab.stoi['<pad>'],
                                      d_k=args.d_k,
                                      d_v=args.d_v,
                                      h=args.head,
                                      d_model=args.d_model
                                      )
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder, args=args).to(device)

    dict_dataset_test = target_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    # Load Checkpoint
    if args.checkpoint_path:
        fname = args.checkpoint_path
    else:
        fname = 'saved_models/%s_best.pth' % args.exp_name

    if os.path.exists(fname):
        print(f"Loading model from {fname}")
        data = torch.load(fname, map_location=device, weights_only=False)
        model.load_state_dict(data['state_dict'], strict=False)
    else:
        print(f"Checkpoint not found at {fname}")
        exit(1)

    # Test scores
    scores = evaluate_metrics(model, dict_dataloader_test, text_field, device)
    
    print("\nEvaluation Results:")
    print(f"CIDEr: {scores['CIDEr']}")
    print(f"BLEU: {scores['BLEU']}")
    print(f"METEOR: {scores['METEOR']}")
    print(f"ROUGE: {scores['ROUGE']}")
