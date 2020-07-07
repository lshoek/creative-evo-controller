from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

from utils import save_im
from utils import load_im

import matplotlib.pyplot as plt
import numpy as np
import asyncio
import collections
import json
import math
import time
import random
import torch
import lz4.frame

OSC_HELLO = "/hi"
OSC_BYE = "/bye"
OSC_INFO = "/info"
OSC_ACTIVATION = "/act"
OSC_FITNESS = "/fit"
OSC_JOINTS = "/jnts"
OSC_PULSE = "/pls"

OSC_ARTIFACT_START = "/art/start/"
OSC_ARTIFACT_PART = "/art/part/"
OSC_ARTIFACT_END = "/art/end/"
OSC_SIZE = 2048

TIMESTEP_EPSILON = 1e-3
TIMESTEP_BOUND = math.pi
TIMESTEP_MAX = TIMESTEP_BOUND-TIMESTEP_EPSILON

def norm01(x):
	return (x+1.0)*0.5

def cpg(t):
	return norm01(math.sin(np.float(t)))

class Client():
	def __init__(self, host=None, inport=None, outport=None, obs_size=64):
		self.action_queue = []
		self.joints_queue = []
		self.obs_queue = []
		self.obs_parts = []
		self.obs_size = obs_size
		self.last_obs = np.zeros((obs_size, obs_size))

		if (host==None or inport==None or outport==None):
			with open('config/osc.json') as f:
				config = json.load(f)
				self.host = config.get('client.host')
				self.inport = int(config.get('client.port'))
				self.outport = int(config.get('server.port'))
		else:
			self.host = host
			self.inport = int(inport)
			self.outport = int(outport)

		self.client = SimpleUDPClient(self.host, self.outport)

		self.handshake = True
		self.finished = False
		self.terminate = False
		self.save_obs = True

		self.fitness = 0
		self.clock = 0.0
		self.oscillator = 0.0

		self.dispatcher = Dispatcher()
		self.dispatcher.map(f'{OSC_HELLO}*', self.__dispatch_hello, self)
		self.dispatcher.map(f'{OSC_BYE}*', self.__dispatch_bye, self)
		self.dispatcher.map(f'{OSC_INFO}*', self.__dispatch_info, self)
		self.dispatcher.map(f'{OSC_FITNESS}*', self.__dispatch_fitness, self)
		self.dispatcher.map(f'{OSC_JOINTS}*', self.__dispatch_joints_packets, self)
		self.dispatcher.map(f'{OSC_ARTIFACT_START}*', self.__dispatch_start_packets, self)
		self.dispatcher.map(f'{OSC_ARTIFACT_PART}*', self.__dispatch_append_packets, self)
		self.dispatcher.map(f'{OSC_ARTIFACT_END}*', self.__dispatch_process_packets, self)


	def __dispatch_hello(self, addr, args, packets=None):
		if self.handshake:
			self.client.send_message(OSC_HELLO, 0)
			self.handshake = False
			self.clock = time.time()

	def __dispatch_bye(self, addr, args, packets=None):
		if self.save_obs:
			save_im(np.reshape(im, (self.obs_size, self.obs_size)))
			#self.__visualize_debug(im)
		self.terminate = True

	def __dispatch_info(self, addr, args, packets=None):
		info = addr.split('/')
		name, num_joints, num_outputs, canvas_height = info[2], int(info[3]), int(info[4]), int(info[5])

		print(f'Connected with {name}: {num_joints} joints, {num_outputs-num_joints} brush(es)')
		if canvas_height != self.obs_size:
			print(f'Error: VAE input size ({canvas_height}) and canvas size ({self.obs_size}) mismatch!')

		self.client.send_message(f'{OSC_INFO}/{self.id}/{self.eval_id}/{self.generation}/{self.time_limit}', 0)

	def __dispatch_fitness(self, addr, args, packets=None):
		self.fitness = packets
		self.finished = True

	def __dispatch_start_packets(self, addr, args, packets=None):
		self.obs_parts = []

	def __dispatch_append_packets(self, addr, args, packets=None):
		self.obs_parts.append(packets)

	def __dispatch_process_packets(self, addr, args, packets=None):
		creature_id = packets
		data = b''.join(self.obs_parts)
		#print(f'[<-] received {len(data)} bytes')

		data_uncomp = lz4.frame.decompress(data)
		im = np.frombuffer(data_uncomp, dtype=np.int8)
		im = np.reshape(im, (1, 1, self.obs_size, self.obs_size))	
		im = im.astype(np.float32)
		im = torch.from_numpy(im)
		self.obs_queue.append((creature_id, im))

	def __dispatch_joints_packets(self, addr, args, packets=None):
		data = np.frombuffer(packets, dtype=np.float32)
		state = torch.from_numpy(data)
		self.joints_queue.append(state)

	def __send_activation(self, msg):
		creature_id, output, pulse = msg
		self.client.send_message(f'{OSC_ACTIVATION}/{creature_id}', output.tobytes())
		self.client.send_message(f'{OSC_PULSE}/{creature_id}', pulse)
		#print(f'[->] sent {len(output.tobytes())} bytes ({output})')

	def __visualize_debug(self, im):
		plt.ion()
		plt.subplot(1, 2, 1), plt.imshow(im)
		plt.pause(0.001)
		plt.show()
		print(im)

	def __activate(self, rollout_gen, obs, bodystate, pulse):		
		pulse_tensor = torch.tensor([pulse])
		action = rollout_gen.get_action(obs, bodystate, pulse_tensor)
		return action

	async def __loop(self, rollout_gen):
		i = 0
		while not self.terminate:
			if not self.handshake:

				# handle message queue
				if len(self.action_queue) > 0:
					for msg in self.action_queue:
						self.__send_activation(msg)
					self.action_queue = []

				# handle observations
				if len(self.obs_queue) > 0 and len(self.joints_queue) > 0:
					creature_id, obs = self.obs_queue.pop()
					bodystate = self.joints_queue.pop()
					action = self.__activate(rollout_gen, obs, bodystate, self.oscillator)

					self.clock += abs(action[-1]) * TIMESTEP_MAX
					self.oscillator = cpg(self.clock)

					self.action_queue.append((creature_id, action[:-1], self.oscillator))
					self.obs_queue = []
					
					i += 1

				# quit when results are in
				if self.finished:
					return

			await asyncio.sleep(0.0)

	async def __start(self, rollout_gen):
		server = AsyncIOOSCUDPServer((self.host, self.inport), self.dispatcher, asyncio.get_event_loop())
		transport, protocol = await server.create_serve_endpoint()

		await self.__loop(rollout_gen)
		transport.close()

	def start(self, generation, id, eval_id, rollout_gen):
		self.time_limit = rollout_gen.time_limit
		self.prev_action = np.random.rand(rollout_gen.output_size).astype(np.float32)
		self.generation = generation
		self.id = id
		self.eval_id = eval_id

		print(f'Started {generation}:{id}')
		asyncio.run(self.__start(rollout_gen))

		print(f'Finished {generation}:{id} / fitness: {self.fitness}')
		return self.fitness
