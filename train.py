import math 
import torch
import copy
import random
import json

import numpy as np
from torchvision import transforms
from multiprocessing import Lock

from models.autoenc import AutoEncoder
from models.controller import Controller
from client import Client

OBS_SIZE = 64
LATENT_SIZE = 32

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.fill_(0.01)

class RolloutGenerator(object):
    def __init__(self, device, time_limit):

        with open('conf/creature.json') as f:
            config = json.load(f)
            self.obs_size = config.get('obs.size')
            self.num_joints = config.get('joints.size')
            self.num_brushes = config.get('brushes.size')
            self.cpg_enabled = config.get('cpg.enabled')

            self.output_size = self.num_joints + self.num_brushes
            self.input_size = self.num_joints + LATENT_SIZE

        self.client = Client(obs_size=OBS_SIZE)

        self.device = device
        self.time_limit = time_limit

        # inputs (add an extra pulse neuron to the controller if cpg is enabled)
        if self.cpg_enabled:
            self.input_size = self.input_size + 1
            self.output_size = self.output_size + 1

        # pre-trained auto encoder from disk?
        self.auto = AutoEncoder(LATENT_SIZE)

        # controller is trained on the go
        self.controller = Controller(self.input_size, self.output_size).cuda()
        self.controller.apply(init_weights).cuda()

        # print(self.auto)
        # print(self.controller)

    def get_action(self, obs, bodystate, pulse):
        bodystate_comp = torch.cat((bodystate, pulse)) if self.cpg_enabled else bodystate
        latent = self.auto.cuda().encode(obs.cuda())
        action = self.controller.cuda().forward(latent.flatten(), bodystate_comp.cuda().flatten())
        
        return action.squeeze().cpu().numpy()

    def do_rollout(self, generation, id, eval_id, early_termination=True):
        with torch.no_grad():    
            fitness = self.client.start(generation, id, eval_id, rollout_gen=self)
            return fitness, 0


def fitness_eval_parallel(pool, r_gen, generation, id, eval_id, early_termination=True):
    return pool.apply_async(r_gen.do_rollout, args=(generation, id, eval_id, early_termination) )


class GAIndividual():
    '''
    GA Individual

    multi = flag to switch multiprocessing on or off
    '''
    def __init__(self, device, time_limit, multi=False):
        self.device = device
        self.time_limit = time_limit
        self.multi = multi

        self.mutation_power = 0.01 

        self.rollout_gen = RolloutGenerator(device, time_limit)

        self.async_results = []
        self.calculated_results = {}

    def run_solution(self, pool, generation, local_id, evals=1, early_termination=True, force_eval=False):

        if force_eval:
            self.calculated_results.pop(evals, None)

        if (evals in self.calculated_results.keys()): #Already caculated results
            return

        self.async_results = []

        for i in range(evals):
            if self.multi:
                self.async_results.append(fitness_eval_parallel(pool, generation, local_id, i, self.rollout_gen, early_termination))#, self.controller_parameters) )
            else:
                self.async_results.append(self.rollout_gen.do_rollout(generation, local_id, i, early_termination)) 


    def evaluate_solution(self, evals):

        if (evals in self.calculated_results.keys()): #Already calculated?
            mean_fitness, std_fitness = self.calculated_results[evals]

        else:
            if self.multi:
                results = [t.get()[0] for t in self.async_results]
            else:
                results = [t[0] for t in self.async_results]

            mean_fitness = np.mean(results)
            std_fitness = np.std(results)

            self.calculated_results[evals] = (mean_fitness, std_fitness)

        self.fitness = -mean_fitness

        return mean_fitness, std_fitness


    def load_solution(self, filename):

        s = torch.load(filename)

        self.rollout_gen.auto.load_state_dict( s['vae'])
        self.rollout_gen.controller.load_state_dict( s['controller'])

    
    def clone_individual(self):
        child_solution = GAIndividual(self.device, self.time_limit, multi=True)
        child_solution.multi = self.multi

        child_solution.fitness = self.fitness

        child_solution.rollout_gen.controller = copy.deepcopy (self.rollout_gen.controller)
        child_solution.rollout_gen.auto = copy.deepcopy (self.rollout_gen.auto)
        
        return child_solution
    
    def mutate_params(self, params):
        for key in params: 
            params[key] += torch.from_numpy(np.random.normal(0, 1, params[key].size()) * self.mutation_power).cuda().float()

    def mutate(self):
        self.mutate_params(self.rollout_gen.controller.state_dict())
        #self.mutate_params(self.rollout_gen.auto.state_dict())
