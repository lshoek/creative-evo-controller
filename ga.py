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
from utils import rankmin

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

    def run(self, max_generations, filename, folder):
        best_f = -sys.maxsize

        fitness_file = open(folder+"/fitness_"+filename+".txt", 'a')
        ind_fitness_file = open(folder+"/individual_fitness_"+filename+".txt", 'a')
        
        g = 0
        P = self.P
        pop_name = folder+"/pop_"+filename+".p"

        # Load previously saved population
        if os.path.exists(pop_name):
            pop_tmp = torch.load(pop_name)
            print("Loading existing population ",pop_name, len(pop_tmp))

            idx = 0
            for s in pop_tmp:
                 P[idx].rollout_gen.vae.load_state_dict ( s['vae'].copy() )
                 P[idx].rollout_gen.controller.load_state_dict ( s['controller'].copy() )

                 g = s['generation'] + 1
                 idx+=1

        while g < max_generations: 
            start_time = time.time()
            fitness = np.zeros(self.pop_size)

            print(f'Generation {g}')
            print(f'Evaluating individuals: {len(P)}')

            # ask the ES to give us a set of candidate solutions
            solutions = self.solver.ask()

            # evaluate all candidates
            for i, s in enumerate(P):  
                s.set_controller_params(solutions[i])
                f = s.run_solution(generation=g, local_id=i)
                fitness[i] = f

            current_f = np.max(fitness)
            average_f = np.mean(fitness)
            print(f'Current best: {current_f}\nCurrent average: {average_f}\n All-time best: {best_f}')

            # return rewards to ES for param update
            max_index = np.argmax(fitness)
            fitness = rankmin(fitness)
            self.solver.tell(fitness)
            new_results = self.solver.result()

            # process results
            if current_f > best_f:
                best_controller = solutions[max_index] # SAVE THIS AND THE SOLVER

                # Save solver and change level to a random one
                _dir = os.path.join('saved_models', self.init_time)
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
                
                solver_path = os.path.join(_dir, "solver.pkl")
                pickle.dump(self.solver, open(solver_path, 'wb'))
                best_f = current_f

            save_pop = []
            for s in P:
                ind_fitness_file.write("Gen\t%d\tFitness\t%f\n" % (i, -s.fitness ))  
                ind_fitness_file.flush()

                save_pop += [{
                     'controller': s.rollout_gen.controller.state_dict(), 
                     'fitness':fitness, 'generation':i
                }]
            print("Saving population")
            torch.save(save_pop, folder+"/pop_"+filename+".p")

            elapsed_time = time.time() - start_time

            print("%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\tTime\t%f\n" %                (i, average_f, current_f, best_f, elapsed_time))
            fitness_file.write("%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\tTime\t%f\n" %   (i, average_f, current_f, best_f, elapsed_time))
            fitness_file.flush()

            if (i > max_generations):
                break

            gc.collect()
            g += 1

        fitness_file.close()
        ind_fitness_file.close()
        print('Finished')
   