import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
# from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
# from models.transformer_fix import Transformer, TransformerEncoder, TransformerDecoderLayer, \
#     ScaledDotProductAttention
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import time
from data import build_image_field
from models import model_factory
import spacy
from utils.geo_reward import get_geo_reward

def evaluate_loss(model, dataloader, loss_fn, text_field, e, device):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, boxes, grids, masks, classes, depth_regions, depth_grids, captions) in enumerate(dataloader):
                detections, boxes, grids, masks, classes, depth_regions, depth_grids, captions = detections.to(device), boxes.to(device), grids.to(
                    device), masks.to(device), classes.to(device), depth_regions.to(device), depth_grids.to(device), captions.to(device)

                out = model(detections, boxes, grids, masks, captions, classes=classes)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, e, device):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
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
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field, e, scheduler, device):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, boxes, grids, masks, classes, depth_regions, depth_grids, captions) in enumerate(dataloader):
            detections, boxes, grids, masks, classes, depth_regions, depth_grids, captions = detections.to(device), boxes.to(device), grids.to(
                device), masks.to(device), classes.to(device), depth_regions.to(device), depth_grids.to(device), captions.to(
                device)

            out = model(detections, boxes, grids, masks, captions, classes=classes)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

            loss = loss.mean()
            loss.backward()

            if torch.isnan(loss):
                print('out')
                print(out)
                print('detections:')
                print(detections)
                raise NotImplementedError
            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            # scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field, e, device):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, ((detections, boxes, grids, masks, classes, depth_regions, depth_grids), caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            boxes = boxes.to(device)
            grids = grids.to(device)
            masks = masks.to(device)
            classes = classes.to(device)
            depth_regions = depth_regions.to(device)
            
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size,
                                                **{'boxes': boxes, 'grids': grids, 'masks': masks, 'classes': classes})
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            
            # Safe-DLCT: Geometry-Consistent Reward
            # Expand depths and classes to match beam size
            depth_regions_expanded = depth_regions.repeat_interleave(beam_size, dim=0)
            classes_expanded = classes.repeat_interleave(beam_size, dim=0)
            
            # caps_gen is a dict {i: [cap_str]} from PTBTokenizer
            # Extract strings for get_geo_reward
            caps_gen_list = [caps_gen[i][0] for i in range(len(caps_gen))]
            
            reward_geo = get_geo_reward(caps_gen_list, depth_regions_expanded.cpu().numpy(), classes_expanded.cpu().numpy())
            reward_geo = reward_geo.reshape(detections.shape[0], beam_size)
            
            # Combine rewards (Alpha = 0.5)
            reward = reward.reshape(detections.shape[0], beam_size)
            reward = reward + 0.5 * reward_geo
            
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--grid_on', action='store_true', default=False)
    parser.add_argument('--rl_batch_size', type=int, default=50)
    parser.add_argument('--rl_learning_rate', type=float, default=5e-6)
    parser.add_argument('--max_detections', type=int, default=50)
    parser.add_argument('--dim_feats', type=int, default=2048)
    parser.add_argument('--image_field', type=str, default="ImageDetectionsField")
    parser.add_argument('--model', type=str, default="transformer_fix")
    parser.add_argument('--grid_embed', action='store_true', default=True)
    parser.add_argument('--box_embed', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_last_switch', action='store_true', default=False)
    parser.add_argument('--rl_at', type=int, default=19)

    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'vizwiz'])
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Meshed-Memory Transformer Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    start = time.time()
    image_field = build_image_field(args)

    print('image field time')
    print(time.time() - start)
    start = time.time()
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    print('text field time')
    print(time.time() - start)
    start = time.time()

    # Create the dataset
    if args.dataset == 'coco':
        dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    elif args.dataset == 'vizwiz':
        # args.annotation_folder should point to dataset_vizwiz.json
        # args.features_path should point to directory with HDF5 files
        from data import VizWiz
        # Assuming args.annotation_folder is the directory containing dataset_vizwiz.json
        ann_file = os.path.join(args.annotation_folder, 'dataset_vizwiz.json')
        # Image root for VizWiz is typically data/vizwiz
        img_root = 'data/vizwiz' 
        dataset = VizWiz(image_field, text_field, img_root, ann_file)
        
    print('dataset time')
    print(time.time() - start)
    start = time.time()
    train_dataset, val_dataset, test_dataset = dataset.splits
    print('split time')
    print(time.time() - start)
    start = time.time()

    if args.dataset == 'vizwiz':
        vocab_path = 'vocab_vizwiz.pkl'
        if os.path.isfile(vocab_path):
             print(f"Loading vocabulary from {vocab_path}")
             text_field.vocab = pickle.load(open(vocab_path, 'rb'))
        else:
             print("Building vocabulary for VizWiz")
             text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
             pickle.dump(text_field.vocab, open(vocab_path, 'wb'))
    else:
        if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
            print("Building vocabulary")
            text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
            pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
        else:
            text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    print('build vocab time')
    print(time.time() - start)
    start = time.time()
    # Model and dataloaders
    Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention = model_factory(args)
    encoder = TransformerEncoder(args.n_layer, 0, attention_module=ScaledDotProductAttention,
                                 d_in=args.dim_feats,
                                 d_in_region=1024, # Explicitly set region dim to 1024
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
    print('build model time')
    print(time.time() - start)
    start = time.time()
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = None
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    print('prepare dataset time')
    print(time.time() - start)


    def lambda_lr(s):
        base_lr = 0.0001
        if s <= 3:
            lr = base_lr * s / 4
        elif s <= 10:
            lr = base_lr
        elif s <= 12:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        # s += 1
        return lr


    def lambda_rl_lr(s):
        base_lr = args.rl_learning_rate
        # print('call lambda rl lr s=',s)
        if s <= args.rl_at + 10:
            lr = base_lr
        else:
            lr = base_lr * 0.1
        # elif s <= 12:
        #     lr = base_lr * 0.2
        # else:
        #     lr = base_lr * 0.2 * 0.2
        # s += 1
        return lr


    # Initial conditions
    optim = Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98))
    # scheduler = LambdaLR(optim, lambda_lr)
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])  # ,reduction='none')
    # loss_fn = LabelSmoothing(0.1, ignore_index=text_field.vocab.stoi['<pad>'])  # ,reduction='none')
    use_rl = False
    best_cider = .0
    best_val_loss = float('inf')
    patience = 0
    start_epoch = 0
    switch_to_rl = False

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname, weights_only=False)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])

            if switch_to_rl and not best and not args.use_last_switch:
                data = torch.load('saved_models/%s_best.pth' % args.exp_name, weights_only=False)
                torch.set_rng_state(data['torch_rng_state'])
                torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_val_loss = data.get('best_val_loss', float('inf'))
            patience = data['patience']
            use_rl = data['use_rl']
            if use_rl:
                scheduler = LambdaLR(optim, lambda_rl_lr)  # parameter from self-critical
            else:
                scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5)
            scheduler.load_state_dict(data['scheduler'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

            # Force RL mode if resuming at or after rl_at
            if start_epoch >= args.rl_at and not use_rl:
                print(f"Forcing RL mode: start_epoch ({start_epoch}) >= rl_at ({args.rl_at})")
                use_rl = True
                # Re-initialize optimizer and scheduler for RL
                optim = Adam(model.parameters(), lr=1)
                scheduler = LambdaLR(optim, lambda_rl_lr)
                for i in range(start_epoch):
                    scheduler.step()

    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.rl_batch_size // 5, shuffle=True,
                                       num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.rl_batch_size // 5)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.rl_batch_size // 5)
    print(f"Training starts. Train loader len: {len(dataloader_train)}")
    for e in tqdm(range(start_epoch, start_epoch + 57), desc='Epochs'):
        try:
            if not use_rl:
                train_loss = train_xe(model, dataloader_train, optim, text_field, e, scheduler, device)
                writer.add_scalar('data/train_loss', train_loss, e)
            else:
                if cider_train is None:
                    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
                train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train,
                                                                 text_field, e, device)
                writer.add_scalar('data/train_loss', train_loss, e)
                writer.add_scalar('data/reward', reward, e)
                writer.add_scalar('data/reward_baseline', reward_baseline, e)

            # Validation loss
            val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, e, device)
            writer.add_scalar('data/val_loss', val_loss, e)

            # Validation scores
            scores = evaluate_metrics(model, dict_dataloader_val, text_field, e, device)
            print("Validation scores", scores)
            val_cider = scores['CIDEr']
            writer.add_scalar('data/val_cider', val_cider, e)
            writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
            writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
            writer.add_scalar('data/val_meteor', scores['METEOR'], e)
            writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

            current_lr = optim.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('data/learning_rate', current_lr, e)

            if use_rl:
                scheduler.step()
            else:
                scheduler.step(val_loss)

            # print(scheduler.state_dict().keys())
            # print('step count',scheduler.state_dict()['_step_count'])
            # print('base lrs', scheduler.state_dict()['base_lrs'])
            # print('last lr',scheduler.state_dict()['_last_lr'])
            # print('get lr called withi step',scheduler.state_dict()['_get_lr_called_within_step'])


            # Test scores
            scores = evaluate_metrics(model, dict_dataloader_test, text_field, e, device)
            print("Test scores", scores)
            test_cider = scores['CIDEr']
            writer.add_scalar('data/test_cider', scores['CIDEr'], e)
            writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
            writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
            writer.add_scalar('data/test_meteor', scores['METEOR'], e)
            writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

            # Prepare for next epoch
            best = False
            if val_cider >= best_cider:
                best_cider = val_cider
                patience = 0
                best = True
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best = True
                
            if not best:
                patience += 1


            switch_to_rl = False
            exit_train = False
            if e == args.rl_at:
                if not use_rl:
                    use_rl = True
                    switch_to_rl = True
                    patience = 0
                    optim = Adam(model.parameters(), lr=1)
                    scheduler = LambdaLR(optim, lambda_rl_lr)  # parameter from self-critical
                    for i in range(e):
                        scheduler.step()
                    print("Switching to RL")
            if patience >= 10:
                if not use_rl:
                    use_rl = True
                    switch_to_rl = True
                    patience = 0
                    optim = Adam(model.parameters(), lr=1)
                    scheduler = LambdaLR(optim, lambda_rl_lr)  # parameter from self-critical
                    for i in range(e):
                        scheduler.step()
                    print("Switching to RL")
                elif patience == 10:
                    print('patience reached.')
                    exit_train = True

            if switch_to_rl and not best and not args.use_last_switch:
                data = torch.load('saved_models/%s_best.pth' % args.exp_name, weights_only=False)
                torch.set_rng_state(data['torch_rng_state'])
                torch.cuda.set_rng_state(data['cuda_rng_state'])
                np.random.set_state(data['numpy_rng_state'])
                random.setstate(data['random_rng_state'])
                model.load_state_dict(data['state_dict'])
                print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                    data['epoch'], data['val_loss'], data['best_cider']))

            torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'val_loss': val_loss,
                'val_cider': val_cider,
                'best_val_loss': best_val_loss,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'patience': patience,
                'best_cider': best_cider,
                'use_rl': use_rl,
            }, 'saved_models/%s_last.pth' % args.exp_name)

            if best:
                copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)


            if switch_to_rl:
                copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_xe_res.pth' % args.exp_name)

            if exit_train:
                writer.close()
                break
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"Error in epoch loop: {ex}")
            exit(1)


