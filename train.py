import math 
import torch
import copy
import random
import json

import numpy as np
from torchvision import transforms
from multiprocessing import Lock

from models.vae import VAE
from models.controller import Controller
from client import Client

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.fill_(0.01)

class RolloutGenerator(object):
    def __init__(self, ga):
        # ga reference
        self.ga = ga

        # compressor model
        self.vae = ga.compressor

        # controller model; trained on the go
        self.controller = Controller(ga.input_size, ga.output_size).cuda()
        self.controller.apply(init_weights).cuda()

    def get_action(self, obs, bodystate, brushstate, pulse):
        bodystate_comp = torch.cat((bodystate, brushstate, pulse)) if self.ga.cpg_enabled else torch.cat((bodystate, brushstate))
        latent_mu, _ = self.vae.cuda().encoder(obs.cuda())
        action = self.controller.cuda().forward(latent_mu.flatten(), bodystate_comp.cuda().flatten())

        return action.squeeze().cpu().numpy()

    def do_rollout(self, generation, id, early_termination=True):
        with torch.no_grad():  
            client = Client(self.ga.obs_size)
            return client.start(generation, id, rollout=self)


class GAIndividual():
    def __init__(self, init_time, input_size, output_size, obs_size, compressor, cpg_enabled, device, time_limit):
        self.init_time = init_time
        self.input_size = input_size
        self.output_size = output_size
        self.obs_size = obs_size
        self.compressor = compressor

        self.cpg_enabled = cpg_enabled
        self.device = device
        self.time_limit = time_limit

        self.mutation_power = 0.01
        self.elite = False 
        self.id = 0

        self.fitness = 0.0

        self.rollout_gen = RolloutGenerator(self)
        self.calculated_results = {}

    def run_solution(self, generation, local_id, early_termination=True):
        self.id = local_id
        self.fitness = self.rollout_gen.do_rollout(generation, local_id, early_termination)
        return self.fitness

    def load_solution(self, filename):
        s = torch.load(filename)
        self.rollout_gen.vae.load_state_dict( s['vae'])
        self.rollout_gen.controller.load_state_dict( s['controller'])

    def clone_individual(self):
        child_solution = GAIndividual(self.init_time, self.input_size, self.output_size, self.obs_size, self.rollout_gen.vae, self.cpg_enabled, self.device, self.time_limit)
        child_solution.fitness = self.fitness
        child_solution.rollout_gen.controller = copy.deepcopy(self.rollout_gen.controller)
        
        return child_solution
    
    def mutate_params(self, params):
        for key in params: 
            params[key] += torch.from_numpy(np.random.normal(0, 1, params[key].size()) * self.mutation_power).cuda().float()

    def mutate(self):
        self.mutate_params(self.rollout_gen.controller.state_dict())

    def set_controller_params(self, params):
        new_params = torch.tensor(params, dtype=torch.float32).cuda()
        params = self.rollout_gen.controller.state_dict()   
        shape = params['fc.weight'].shape
        params['fc.weight'].data.copy_(new_params.view(shape))
        return
