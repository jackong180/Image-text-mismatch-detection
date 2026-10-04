"""Evaluate COSMOS image-caption1-caption2 triples; label 1 means OOC."""
import argparse
import json
from engine.evaluator import evaluate

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--test-json',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',default='outputs/test')
    p.add_argument('--device',default='auto')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threshold',type=float,default=.5,help='Fix before looking at test labels')
    args = p.parse_args()
    result = evaluate(args.checkpoint,args.test_json,args.data_root,args.output,args.device,args.batch_size,args.workers,args.threshold)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
