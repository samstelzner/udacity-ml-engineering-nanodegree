import argparse
import torch
import os

if __name__=='__main__':
    # default to the value in environment variable `SM_MODEL_DIR`. Using args makes the script more portable.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args, _ = parser.parse_known_args()
    
    with open(os.path.join(args.model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))