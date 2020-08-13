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
OSC_END_ROLLOUT = "/end"
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

from enum import Enum
class ClientType(Enum):
	ROLLOUT = 0
	REQUEST = 1

def norm01(x):
	return (x+1.0)*0.5

def cpg(t):
	return norm01(math.sin(np.float(t)))

class Client():
	def __init__(self, client_type, host=None, inport=None, outport=None, obs_size=64):
		self.type = client_type
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

		self.fitness = 0.0
		self.clock = 0.0
		self.oscillator = 0.0
		self.brush = 0.5

		self.dispatcher = Dispatcher()

		if client_type == ClientType.ROLLOUT:
			self.dispatcher.map(f'{OSC_HELLO}*', self.__dispatch_hello)
			self.dispatcher.map(f'{OSC_BYE}*', self.__dispatch_bye)
			self.dispatcher.map(f'{OSC_INFO}*', self.__dispatch_info)
			self.dispatcher.map(f'{OSC_END_ROLLOUT}*', self.__dispatch_end)

			self.dispatcher.map(f'{OSC_JOINTS}*', self.__dispatch_joints_packets)
			self.dispatcher.map(f'{OSC_ARTIFACT_START}*', self.__dispatch_start_packets)
			self.dispatcher.map(f'{OSC_ARTIFACT_PART}*', self.__dispatch_append_packets)
			self.dispatcher.map(f'{OSC_ARTIFACT_END}*', self.__dispatch_process_packets)
		else:
			self.dispatcher.map(f'{OSC_FITNESS}*', self.__dispatch_fitness)


	def __dispatch_hello(self, addr, packets=None):
		if self.handshake:
			self.client.send_message(OSC_HELLO, 0)
			self.handshake = False
			self.clock = time.time()

	def __dispatch_bye(self, addr, packets=None):
		self.terminate = True

	def __dispatch_info(self, addr, packets=None):
		info = addr.split('/')
		name, num_joints, num_outputs, canvas_height = info[2], int(info[3]), int(info[4]), int(info[5])

		print(f'Connected with {name}: {num_joints} joints, {num_outputs-num_joints} brush(es)')
		if canvas_height != self.obs_size:
			print(f'Error: VAE input size ({canvas_height}) and canvas size ({self.obs_size}) mismatch!')

		self.client.send_message(f'{OSC_INFO}/{self.ga_id}/{self.id}/{self.generation}/{self.time_limit}', 0)

	def __dispatch_end(self, addr, packets=None):
		if self.save_obs:
			im = np.reshape(self.last_obs.detach().numpy(), (64, 64))*255.0
			save_im(im)
			#self.__visualize_debug(im)
		self.finished = True
	
	from typing import List, Any
	def __dispatch_fitness(self, addr:str, *args: List[Any]):
		info = addr.split('/')
		numEntries = int(info[2])
		numStats = int(info[3])
		f = np.reshape(args, (numEntries, numStats))
		self.fitness = f
		self.finished = True

	def __dispatch_start_packets(self, addr, packets=None):
		self.obs_parts = []

	def __dispatch_append_packets(self, addr, packets=None):
		self.obs_parts.append(packets)

	def __dispatch_process_packets(self, addr, packets=None):
		creature_id = packets
		data = b''.join(self.obs_parts)
		#print(f'[<-] received {len(data)} bytes')

		data_uncomp = lz4.frame.decompress(data)
		im = np.frombuffer(data_uncomp, dtype=np.uint8)
		im = np.reshape(im, (1, 1, self.obs_size, self.obs_size))	
		im = im.astype(np.float32)/255.0
		im = torch.from_numpy(im)
		self.obs_queue.append((creature_id, im))
		self.last_obs = im

	def __dispatch_joints_packets(self, addr, packets=None):
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

	def __activate(self, rollout, obs, bodystate, brush, pulse):		
		pulse_tensor = torch.tensor([pulse])
		brush_tensor = torch.tensor([brush])
		action = rollout.get_action(obs, bodystate, brush_tensor, pulse_tensor)
		return action

	async def __loop(self, rollout):
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
					action = self.__activate(rollout, obs, bodystate, self.brush, self.oscillator)

					self.clock += abs(action[-1]) * TIMESTEP_MAX
					self.oscillator = cpg(self.clock)
					self.brush = action[-2]

					self.action_queue.append((creature_id, action[:-1], self.oscillator))
					self.obs_queue = []
					
					i += 1

				# quit when results are in
				if self.finished:
					return

			await asyncio.sleep(0.0)

	async def __request(self):
		self.client.send_message(f'{OSC_FITNESS}', 0)
		while not self.terminate:
			#if not self.handshake:
			if self.finished:
				return

			await asyncio.sleep(0.0)

	async def __start(self, rollout=None):
		server = AsyncIOOSCUDPServer((self.host, self.inport), self.dispatcher, asyncio.get_event_loop())
		transport, protocol = await server.create_serve_endpoint()

		if self.type == ClientType.ROLLOUT:
			await self.__loop(rollout)
		else:
			await self.__request()

		transport.close()

	def start(self, generation=0, id=0, rollout=None):
		if (self.type == ClientType.ROLLOUT):
			self.ga_id = rollout.ga.init_time
			self.time_limit = rollout.ga.time_limit
			self.prev_action = np.random.rand(rollout.ga.output_size).astype(np.float32)
			self.generation = generation
			self.id = id
			print(f'Started {generation}:{id}')
		
		asyncio.run(self.__start(rollout))

		if (rollout): 
			print(f'Finished {generation}:{id}')

		return self.fitness
