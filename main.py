import argparse
import json
import sys
import os
import torch
import random
import time

from ga import GA

torch.set_num_threads(1)
torch.set_printoptions(precision=4)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--client_host', type=str, default='127.0.0.1')
    parser.add_argument('--client_port', type=int, default=1024)

    parser.add_argument('--server_host', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=int, default=1025)

    parser.add_argument('--pop_size', type=int, default=200,
        help='Population size.')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')

    parser.add_argument('--generations', type=int, default=1000, metavar='N',
        help='number of generations to train (default: 1000)')

    parser.add_argument('--threads', type=int, default=4, metavar='N',
        help='threads')

    parser.add_argument('--test', type=str, default='', metavar='N',
        help='0 = no protection, 1 = protection')

    parser.add_argument('--folder', type=str, default='results/sim', metavar='N',
        help='folder to store results')

    parser.add_argument('--top', type=int, default=3, metavar='N',
        help='numer of top elites that should be re-evaluated')
    
    parser.add_argument('--elite_evals', type=int, default=20, metavar='N',
        help='how many times should the elite be evaluated')                        

    parser.add_argument('--timelimit', type=int, default=1000, metavar='N',
        help='time limit per evaluation')

    return parser.parse_args()

def main(argv):
    args = parse_args()
    if (args.pop_size % 2 != 0):
        print("Error: Population size needs to be an even number.")
        exit()

    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    #seed = random.randint(0, 16384)
    random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ga = GA(args.elite_evals, args.top, args.threads, args.timelimit, args.pop_size, device)
    ga.run(args.generations, f'{time.time()}', args.folder)

if __name__ == '__main__':
    main(sys.argv)
