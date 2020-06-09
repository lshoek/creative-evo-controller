from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

from utils import save_im
from utils import load_im
from autoenc import SketchAutoEncoder

import matplotlib.pyplot as plt
import numpy as np
import asyncio
import imageio
import math
import lz4.frame

OSC_FRAME_START = "/frame/start/"
OSC_FRAME_PART = "/frame/part/"
OSC_FRAME_END = "/frame/end/"
OSC_SIZE = 2048

def start_handler(addr, args, packets=None):
	args[0].clear_parts()

def start_handler(addr, args, packets=None):
	args[0].clear_parts()

def part_handler(addr, args, packets=None):
	args[0].append_part(packets)

def end_handler(addr, args, compression_flag=1):
	args[0].process_msg(compression_flag)

class Client():
	def __init__(self, host, inport, outport):
		self.parts = []
		self.queue = []
		
		self.host = host
		self.inport = int(inport)
		self.outport = int(outport)

		self.client = SimpleUDPClient(self.host, self.outport)

		self.dispatcher = Dispatcher()
		self.dispatcher.map(f'{OSC_FRAME_START}*', start_handler, self)
		self.dispatcher.map(f'{OSC_FRAME_PART}*', part_handler, self)
		self.dispatcher.map(f'{OSC_FRAME_END}*', end_handler, self)

		self.autoencoder = SketchAutoEncoder()

	def append_part(self, part):
		self.parts.append(part)

	def clear_parts(self):
		self.parts = []

	def process_msg(self, compression_flag):
		data = b''.join(self.parts)

		if (compression_flag == 0):
			im = np.frombuffer(data, dtype=np.int8)
		elif (compression_flag == 1):
			data_uncomp = lz4.frame.decompress(data)
			im = np.frombuffer(data_uncomp, dtype=np.int8)

		im = np.reshape(im, (16, 16))
		im = im.astype(np.float32)

		latent = self.autoencoder.encode(im)
		self.queue.append(latent)
		print(f'[<-] received {len(data)} bytes')

		#im = np.reshape(im, (16, 16))
		#save_im(im)
		#self.__visualize_debug(im)

	def send(self, msg):
		subs = math.ceil(len(msg) / OSC_SIZE)
		self.client.send_message(OSC_FRAME_START, 0)

		for p in range(subs):
			s = p * OSC_SIZE
			e = s + OSC_SIZE
			part = msg[s:e]
			self.client.send_message(f'{OSC_FRAME_PART}{p}', part.numpy().tobytes())
		
		self.client.send_message(OSC_FRAME_END, 0)
		print(f'[->] sent {len(msg)} bytes')

	def __visualize_debug(self, im):
		plt.ion()
		plt.subplot(1, 2, 1), plt.imshow(im)
		plt.pause(0.001)
		plt.show()
		print(im)
	
	async def __loop(self):
		while True:
			if len(self.queue) > 0:
				for msg in self.queue:
					self.send(msg)
				self.queue = []
			
			await asyncio.sleep(0.0)

	async def __start(self):

		server = AsyncIOOSCUDPServer((self.host, self.inport), self.dispatcher, asyncio.get_event_loop())

		# create datagram endpoint and start serving
		transport, protocol = await server.create_serve_endpoint()

		print(f'sending to {self.host}:{self.outport}...')
		print(f'listening on {self.host}:{self.inport}...')

		# enter main
		await self.__loop()

		# clean up serve endpoint
		transport.close()

	def init(self):
		asyncio.run(self.__start())
