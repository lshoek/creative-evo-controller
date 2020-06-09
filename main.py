import argparse
import json
import os
import asyncio
from client import Client

def parse_args():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--ip', type=str)
	argparser.add_argument('--port', type=int)
	return argparser.parse_args()

# main()
if __name__ == '__main__':

	args = parse_args()
	if args.ip is not None:
		host, inport = args.host, args.port
	else:
		with open('config.json') as f:
			config = json.load(f)
			host = config.get('client.host')
			inport, outport = config.get('client.port'), config.get('server.port')

	client = Client(host, inport, outport)
	client.init()
