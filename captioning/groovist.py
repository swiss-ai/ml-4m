import argparse
import clip
import configparser
import json
import numpy as np
import pandas as pd
from statistics import fmean
import sys
import torch
from torch.nn.functional import normalize
from PIL import Image
import utils


config = configparser.ConfigParser()
config.read('config.ini')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'running on {device}\n')

# compute reverse groovist score for given sid (story id)
def reverse_groovist(sid, iids, preprocess, model, IRs_df, IRs_path):
    IRs = []
    IR_descriptions = []
    for idx in range(len(iids)):
        iid = iids[idx]
        try:
            top_B_IRs = IRs_df[(IRs_df['image_id'] == iid) & (IRs_df['score'] > 0.5)] #.iloc[:B]
            if top_B_IRs.empty:
                top_B_IRs = IRs_df[IRs_df['image_id'] == iid]
            # assert(not top_B_IRs.empty)
            for index, row in top_B_IRs.iterrows():
                IR_box = row['bbox']
                IR_obj = row['object']
                IR_name = str(iid) + '_' + IR_box + '.jpg'
                IR_descriptions.append(IR_obj + "_" + IR_name)
                IRs.append(preprocess(Image.open(IRs_path + '/' + IR_name).convert('RGB')).unsqueeze(0).to(device))
        except Exception as e:
            print(f'image {iid} not used for alignment', e)
    with torch.no_grad():
        IRs_embs = normalize(model.encode_image(torch.stack(IRs).squeeze(1)), p=2, dim=-1)

    alignment_scores_rev = torch.zeros((len(iids), IRs_embs.shape[0]))
    for idx in range(len(iids)):
        NPs_rev, _ = utils.get_concreteness_ratings(nphrases[sid][idx])
        if len(NPs_rev) == 0:
            continue
        NPs_tokenized_rev = clip.tokenize(NPs_rev).to(device)
        with torch.no_grad():
            NPs_embs_rev = normalize(model.encode_text(NPs_tokenized_rev), p=2, dim=-1)
        
        alignment_matrix_rev = IRs_embs @ NPs_embs_rev.T
        alignment_scores_rev[idx] = torch.max(alignment_matrix_rev, dim=1)[0].cpu()

    max_alignments_rev = torch.max(alignment_scores_rev, dim=0)[0]

    return max_alignments_rev, IR_descriptions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='VIST',
                        choices=['VIST', 'AESOP', 'VWP', 'custom'], help='dataset to score')
    parser.add_argument('--input_file', default='data/sample_nphrases.json',
                        help='path to file with NPs (?.json)')
    parser.add_argument('--output_file', default='data/sample_scores.json',
                        help='path to file with GROOViST scores (?.json)')
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as fh:
            nphrases = json.load(fh)
        fh.close()
        print(f'loaded NPs for {len(nphrases)} stories in {args.input_file}\n')
    except Exception as e:
        print(f'unable to read {args.input_file}', e)
        sys.exit(1)

    try:
        theta = config[args.dataset].getfloat('theta')
        sid_2_iids = config[args.dataset]['sid_2_iids_file']
        sid_2_iids = None if sid_2_iids == 'None' else json.load(open(sid_2_iids, 'r'))
        IRs_df = pd.read_csv(config[args.dataset]['image_regions_info_file'])
        IRs_path = config[args.dataset]['image_regions']
        print(f'loaded {args.dataset} dataset configuration\n')
    except Exception as e:
        print(f'unable to read configuration for {args.dataset}', e)
        sys.exit(1)

    try:
        print('bootstrapping the CLIP (ViT-B/32) model...', end='')
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        model = model.to(device)
        print('complete\n')
    except Exception as e:
        print('unable to load the CLIP model', e)
        sys.exit(1)

    # count = 0
    sids = list(nphrases.keys())
    all_scores = []
    for sid in sids:
        # if count == 20:
        #     break
        # count += 1
        print(f'evaluating {sid}...')

        # get all image regions corresponding to story id sid
        iids = utils.get_image_ids(args.dataset, sid, sid_2_iids)

        # reverse Groovist
        max_alignments_rev, IR_descriptions = reverse_groovist(sid, iids, preprocess, model, IRs_df, IRs_path)
        
        try:
            NPs, cr_weights = utils.get_concreteness_ratings([item for row in nphrases[sid] for item in row])
        except Exception as e:
            print("Error in data format. Your noun phrases are probably formatted as <image-sequence, story> pairs.")
            print(f"Error message: {e}")

        NPs_tokenized = clip.tokenize(NPs).to(device)
        with torch.no_grad():
            NPs_embs = normalize(model.encode_text(NPs_tokenized), p=2, dim=-1)

        try:
            # original groovist
            NPs_max_alignment = utils.get_max_alignment_scores(NPs_embs, iids, preprocess, model, IRs_df, IRs_path)
            
            original_score, original_score_weighted = utils.penalize_concretize_normalize_old(NPs_max_alignment, torch.tensor(cr_weights), theta, NPs)
            reverse_score = utils.penalize_concretize_normalize(max_alignments_rev, theta, IR_descriptions)
            original_normalized = np.tanh(original_score_weighted)
            reverse_normalized = np.tanh(reverse_score)
            combined_score = 2*original_normalized*reverse_normalized / (original_normalized + reverse_normalized)

            all_scores.append({'Original': original_score, 'Original weighted': original_score_weighted, 'Reverse': reverse_score, 'Combined': combined_score,
                               'Original Normalized': original_normalized, 'Reverse Normalized': reverse_normalized, \
                                'Combined Normalized': np.tanh(combined_score)})

            print(f'Combined GROOViST score for {sid}: {combined_score}, in range [-1, 1]: {np.tanh(combined_score)}\n')
        except Exception as e:
            print(f'{sid} not used for computing GROOViST', e)
            all_scores.append({'Original': 0.000, 'Original weighted': 0.000,'Reverse': 0.000, 'Combined': 0.000, 'Original Normalized': 0.000, 
                               'Reverse Normalized': 0.000, 'Combined Normalized': 0.000})

    with open(args.output_file, 'w') as fh:
        json.dump(dict(zip(sids, all_scores)), fh, indent=2)
    fh.close()
    print(f'saved scores to {args.output_file}\n')
