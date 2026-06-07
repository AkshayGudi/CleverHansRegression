# import torch, numpy as np, cv2, os, csv, json, argparse
# from insight_training.model import construct_PPNet
# from define_parameters import NetworkParams

# '''
# This script analyzes the importance of prototypes for the classification task.
# It computes the importance of each prototype for the classification task and plots the results.

# Usage:
# python3 analyze_proto_importance.py \
#   --ckpt .../saved_models/Epoch_50_after_protopushing.pth \
#   --test_dir /path/to/data_store/DR/bld_artifact/class3_v14/test \
#   --artifact_csv /path/to/data_store/DR/bld_artifact/class3_v14/data_details_class3/test_labeled_data.csv \
#   --test_config config/datasplit/dr_config/dr_test_config.json \
#   --target_class 3
# '''

# parser = argparse.ArgumentParser()
# parser.add_argument('--ckpt', required=True)
# parser.add_argument('--test_dir', required=True)
# parser.add_argument('--artifact_csv', required=True, help='CSV with image_name,artifact_label')
# parser.add_argument('--test_config', required=True, help='dr_test_config.json')
# parser.add_argument('--target_class', type=str, default='3')
# parser.add_argument('--n_samples', type=int, default=30)
# args = parser.parse_args()

# ppnet = construct_PPNet(network_params=NetworkParams())
# sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
# ppnet.load_state_dict(sd, strict=True); ppnet.eval()

# with open(args.artifact_csv) as f:
#     art = {r['image_name']+'.jpeg' for r in csv.DictReader(f) if r['artifact_label']=='1'}
# tcfg = json.load(open(args.test_config))
# lab = {n+'.jpeg': l for n, l in zip(tcfg['files'], tcfg['labels'])}

# pos = [n for n in art if lab.get(n)==args.target_class]
# neg = [n+'.jpeg' for n in tcfg['files'] if lab[n+'.jpeg']!=args.target_class and (n+'.jpeg') not in art]

# def run(names):
#     ims = []
#     for nm in names[:args.n_samples]:
#         im = cv2.imread(os.path.join(args.test_dir, nm))
#         if im is None: continue
#         if im.shape[0]!=540: im = cv2.resize(im,(540,540))
#         # Dimension fix: cv2.imread gives (H, W, C); PyTorch expects
#         # (C, H, W). Previous permute(2, 1, 0) yielded (C, W, H) and
#         # silently transposed spatial axes. Use (2, 0, 1) for HWC -> CHW.
#         ims.append(torch.from_numpy(im/255.).permute(2,0,1).float())
#     with torch.no_grad():
#         _,_,pa = ppnet(torch.stack(ims), return_convs=False)
#     return pa.numpy()

# a_pos, a_neg = run(pos), run(neg)
# c  = sd['proto_classes'].numpy()
# w2 = sd['last_layer.weight'].squeeze().numpy()**2
# m  = w2 / c
# share = lambda a: ((a*m) / (a*m).sum(1,keepdims=True)).mean(0)*100
# s_pos, s_neg = share(a_pos), share(a_neg)
# for k in np.argsort(-(s_pos - s_neg))[:15]:
#     print(f'proto {k:2d}  c={c[k]:.2f}  w²={w2[k]:.2f}  share+={s_pos[k]:.2f}%  share-={s_neg[k]:.2f}%  Δ={s_pos[k]-s_neg[k]:+.2f}%')