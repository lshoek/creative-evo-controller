import sys, random
import numpy as np
import pickle
import torch
import time
import math
import multiprocessing
import gc
import copy
import json
import os
from datetime import datetime
from es import CMAES
from client import Client
from utils import compute_centered_ranks

def set_controller_weights(controller, weights):
    new_params = torch.tensor(weights, dtype=torch.float32).cuda()
    params = controller.state_dict()
    shape = params['fc.weight'].shape
    controller.state_dict()['fc.weight'].data.copy_(new_params.view(shape))
    return

class GA:
    def __init__(self, timelimit, pop_size, device):
        self.pop_size = pop_size
        self.truncation_threshold = int(pop_size/2)  # Should be dividable by two
        self.P = []

        # unique GA id
        self.init_time =  datetime.now().strftime("%Y%m%d_%H%M%S")

        # load configuration params
        with open('config/creature.json') as f:
            config = json.load(f)
            model_fromdisk = config.get('vae.model.fromdisk')
            model_path = config.get('vae.model.path')

            latent_size = config.get('vae.latent.size')
            obs_size = config.get('vae.obs.size')
            num_effectors = config.get('joints.size') + config.get('brushes.size')
            input_size = latent_size + num_effectors
            output_size = num_effectors

            cpg_enabled = config.get('cpg.enabled')
            if cpg_enabled:
                input_size += 1
                output_size += 1

        num_controller_params = input_size * output_size # assuming a single layer
        print(f'Number of controller parameters: {num_controller_params}')

        # load vision module
        from models.vae import VAE
        vae = VAE(latent_size).cuda()

        if model_fromdisk:
            vae.load_state_dict(torch.load(model_path))
            vae.eval() # inference mode
            print(f'Loaded VAE model {model_path} from disk')

        print(f'Generating initial population of {pop_size} candidates...')

        # initialize population
        from train import GAIndividual
        for _ in range(pop_size):
            self.P.append(GAIndividual(
                self.init_time, input_size, output_size, obs_size, 
                compressor=vae, cpg_enabled=cpg_enabled, device=device, time_limit=timelimit))
        
        # initialize cma es
        sigma_init = 4.0
        self.solver = CMAES(num_params=num_controller_params, sigma_init=sigma_init, popsize=pop_size)

    def run(self, max_generations, folder, ga_id=''):
        if (ga_id == ''):
            ga_id = self.init_time
        
        # disk
        results_dir = os.path.join(folder, ga_id)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        fitness_path = os.path.join(results_dir, 'fitness.txt')
        ind_fitness_path = os.path.join(results_dir, 'ind_fitness.txt')
        solver_path = os.path.join(results_dir, "solver.pkl")

        with open(fitness_path, 'a') as file:
            file.write('gen/avg/cur/best\n')

        g = 0
        P = self.P
        pop_name = os.path.join(results_dir, 'population.p')
        best_f = -sys.maxsize

        # initialize controller instance to be saved
        from models.controller import Controller
        best_controller = Controller(P[0].input_size, P[0].output_size)

        # instantiate a separate client to request fitness from simulator
        from client import Client, ClientType

        # Load previously saved population
        if os.path.exists(pop_name):
            pop_tmp = torch.load(pop_name)
            print(f"Loading existing population {pop_name}, {len(pop_tmp)} individuals")

            idx = 0
            for s in pop_tmp:
                P[idx].rollout_gen.controller.load_state_dict(s['controller'].copy())
                g = s['generation'] + 1
                idx+=1

        while g < max_generations: 

            fitness = np.zeros(self.pop_size)
            results_full = np.zeros(self.pop_size)

            print(f'Generation {g}')
            print(f'Evaluating individuals: {len(P)}')

            # ask the ES to give us a set of candidate solutions
            solutions = self.solver.ask()

            # evaluate all candidates
            for i, s in enumerate(P):  
                s.set_controller_weights(solutions[i])
                s.run_solution(generation=g, local_id=i)
            
            # request fitness from simulator
            results_full = Client(ClientType.REQUEST).start()
            fitness = results_full[:,0]

            for i, s in enumerate(P):
                s.fitness = fitness[i]

            current_f = np.max(fitness)
            average_f = np.mean(fitness)
            print(f'Current best: {current_f}\nCurrent average: {average_f}\n All-time best: {best_f}')

            # return rewards to ES for param update
            centered_ranks = compute_centered_ranks(fitness)
            self.solver.tell(centered_ranks)

            max_index = np.argmax(fitness)
            new_results = self.solver.result()
            current_best = new_results[1]

            # process results
            if current_f > best_f:
                set_controller_weights(best_controller, solutions[max_index])
                torch.save(best_controller, os.path.join(results_dir, 'best_controller.pth'))

                # Save solver and change level to a random one
                pickle.dump(self.solver, open(solver_path, 'wb'))
                best_f = current_f

            print("Saving population")
            save_pop = []
            for i, s in enumerate(P):
                
                # /fitness /coverage /coverageReward /IC /PCt0 /PCt1
                res = results_full[i,:]
                res_str = (', '.join(['%.6f']*len(res))) % tuple(res)

                with open(ind_fitness_path, 'a') as file:
                    file.write('Gen\t%d\tId\tResults\t%s\n' % (g, i, res_str))  

                save_pop += [{
                     'controller': s.rollout_gen.controller.state_dict(), 
                     'fitness':fitness, 
                     'generation':g
                }]
            torch.save(save_pop, os.path.join(results_dir, 'population.p'))

            res_str = '%d/%f/%f/%f' % (g, average_f, current_f, best_f)
            print(f'gen/avg/cur/best : {res_str}')
            with open(fitness_path, 'a') as file:
                file.write(f'{res_str}\n')

            if (i > max_generations):
                break

            gc.collect()
            g += 1

        print('Finished')
   