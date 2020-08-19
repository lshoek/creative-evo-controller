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
from client import Client, ClientType

def set_controller_weights(controller, weights):
    params = controller.state_dict()
    shape_weights = params['fc.weight'].shape
    num_bias_weights = len(params['fc.bias'])
    
    new_params = torch.tensor(weights[:-num_bias_weights], dtype=torch.float32).cuda()
    new_params_bias = torch.tensor(weights[-num_bias_weights:], dtype=torch.float32).cuda()

    controller.state_dict()['fc.weight'].data.copy_(new_params.view(shape_weights))
    controller.state_dict()['fc.bias'].data.copy_(new_params_bias)
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

        # report controller parameters
        self.num_controller_params = input_size * output_size + output_size
        print(f'Number of controller parameters: {self.num_controller_params}')


    def run(self, max_generations, folder, ga_id='', init_solution_id=''):
        if (ga_id == ''):
            ga_id = self.init_time
        
        # disk
        results_dir = os.path.join(folder, ga_id)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        fitness_path = os.path.join(results_dir, 'fitness.txt') # most important fitness results per run (for plotting)
        ind_fitness_path = os.path.join(results_dir, 'ind_fitness.txt') # more detailed fitness results per individual
        solver_path = os.path.join(results_dir, "solver.pkl") # contains the current population
        best_solver_path = os.path.join(results_dir, "best_solver.pkl") # contains the current population
        init_solution_path = os.path.join(os.path.join(folder, init_solution_id), "solver.pkl") # path to initial solution solver

        current_generation = 0
        P = self.P
        best_f = -sys.maxsize

        # initialize controller instance to be saved
        from models.controller import Controller
        best_controller = Controller(P[0].input_size, P[0].output_size)

        # initialize cma es (start from scratch or load previously saved solver/population)
        resume = False
        if os.path.exists(solver_path):
            resume = True
            self.solver = pickle.load(open(solver_path, 'rb'))
            new_results = self.solver.result()
            best_f = new_results[1]

            if os.path.exists(fitness_path):
                with open(fitness_path, 'r') as f:
                    lines = f.read().splitlines()
                    last_line = lines[-1]
                    current_generation = int(last_line.split('/')[0])
        # start from scratch but with an initial solution param            
        elif os.path.exists(init_solution_path):
            tmp_solver = pickle.load(open(init_solution_path, 'rb'))
            self.solver = CMAES(num_params=self.num_controller_params, solution_init=tmp_solver.best_param(), sigma_init=0.1, popsize=self.pop_size)
        # completely start from scratch
        else:
            self.solver = CMAES(num_params=self.num_controller_params, sigma_init=0.1, popsize=self.pop_size)
        
        if not resume:
            with open(fitness_path, 'a') as file:
                file.write('gen/avg/cur/best\n')
            with open(ind_fitness_path, 'a') as file:
                file.write('gen/id/fitness/coverage/coverageReward/IC/PC/PCt0/PCt1\n')

        while current_generation < max_generations: 

            fitness = np.zeros(self.pop_size)
            results_full = np.zeros(self.pop_size)

            print(f'Generation {current_generation}')
            print(f'Evaluating individuals: {len(P)}')

            # ask the ES to give us a set of candidate solutions
            solutions = self.solver.ask()

            # evaluate all candidates
            for i, s in enumerate(P):  
                set_controller_weights(s.controller, solutions[i])
                s.run_solution(generation=current_generation, local_id=i)
            
            # request fitness from simulator
            results_full = Client(ClientType.REQUEST).start()
            fitness = results_full[:,0]

            for i, s in enumerate(P):
                s.fitness = fitness[i]

            current_f = np.max(fitness)
            average_f = np.mean(fitness)
            print(f'Current best: {current_f}\nCurrent average: {average_f}\nAll-time best: {best_f}')

            # return rewards to ES for param update
            self.solver.tell(fitness)

            max_index = np.argmax(fitness)
            new_results = self.solver.result()

            # process results
            pickle.dump(self.solver, open(solver_path, 'wb'))
            if current_f > best_f:
                set_controller_weights(best_controller, solutions[max_index])
                torch.save(best_controller, os.path.join(results_dir, 'best_controller.pth'))

                # Save solver and change level to a random one
                pickle.dump(self.solver, open(best_solver_path, 'wb'))
                best_f = current_f

            for i, s in enumerate(P):
                # fitness/coverage/coverageReward/IC/PC/PCt0/PCt1
                res = results_full[i,:]
                res_str = ('/'.join(['%.6f']*len(res))) % tuple(res)

                with open(ind_fitness_path, 'a') as file:
                    file.write('%d/%d/%s\n' % (current_generation, i, res_str))  

            res_str = '%d/%f/%f/%f' % (current_generation, average_f, current_f, best_f)
            print(f'gen/avg/cur/best : {res_str}')
            with open(fitness_path, 'a') as file:
                file.write(f'{res_str}\n')

            if (i > max_generations):
                break

            gc.collect()
            current_generation += 1

        print('Finished')
   