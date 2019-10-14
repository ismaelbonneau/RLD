import matplotlib
matplotlib.use("TkAgg")

import gym
import gridworld
from gym import wrappers, logger
import numpy as np

import Agent

if __name__ == '__main__':

	env = gym.make("gridworld-v0")
	env.seed(0)  # Initialise le seed du pseudo-random
	env.render()  # permet de visualiser la grille du jeu 
	env.render(mode="human") #visualisation sur la console

	# Faire un fichier de log sur plusieurs scenarios
	outdir = 'gridworld-logs/Qlearning-agent-results'
	envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
	env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
	env.seed()  # Initialiser le pseudo aleatoire

	FPS = 0.0001
	obs = envm.reset()
	env.verbose = True

	rsum = 0
	reward = 0
	done = False

	agent = Agent.Qagent(env, gamma=0.5, alpha=0.01)

	j = 0

	nbruns = 50

	for i in range(nbruns):

		obs = envm.reset()
		rsum = 0
		reward = 0
		done = False
		j = 0

		while True:

			action = agent.act(obs, reward, done)
			obs, reward, done, _ = envm.step(action)
			agent.improve(obs, reward, done)
			rsum += reward
			j += 1
			if i % 10 == 0:
				env.render(FPS)

			if done:
				print("Fini! rsum=" + str(rsum) + ", au bout de " + str(j) + " actions")
				break
	env.close()

			