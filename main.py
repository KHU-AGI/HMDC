from method import get_method
from utils.args import parser
import torch
import multiprocessing as mp

import torch._dynamo
torch._dynamo.reset()

def main():
    args = parser.parse_args()
    print(args)

    method = get_method(args.method)(args)
    method.run()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    torch.set_float32_matmul_precision('high')
    main()