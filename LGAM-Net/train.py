"""LGAM-Net weakly supervised training. Run from the project root."""
import argparse
from engine.trainer import train
from utils.runtime import read_config

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cosmos.json')
    p.add_argument('--resume',help='Resume a trusted last.pt checkpoint at an epoch boundary')
    args = p.parse_args()
    train(read_config(args.config),args.resume)

if __name__ == '__main__':
    main()
