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
from client import Client, ClientType

class RolloutGenerator(object):
    def __init__(self, ga):
        # ga reference
        self.ga = ga

        # compressor model
        self.vae = ga.compressor

        # controller model; trained on the go
        self.controller = Controller(ga.input_size, ga.output_size).cuda()

    def get_action(self, obs, bodystate, brushstate, pulse):
        bodystate_comp = torch.cat((bodystate, brushstate, pulse)) if self.ga.cpg_enabled else torch.cat((bodystate, brushstate))
        latent_mu, _ = self.vae.cuda().encoder(obs.cuda())
        action = self.controller.cuda().forward(latent_mu.flatten(), bodystate_comp.cuda().flatten())

        return action.squeeze().cpu().numpy()

    def do_rollout(self, generation, id, early_termination=True):
        with torch.no_grad():  
            client = Client(ClientType.ROLLOUT, self.ga.obs_size)
            client.start(generation, id, rollout=self)


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
        self.controller = self.rollout_gen.controller
        self.calculated_results = {}

    def run_solution(self, generation, local_id, early_termination=True):
        self.id = local_id
        self.rollout_gen.do_rollout(generation, local_id, early_termination)
