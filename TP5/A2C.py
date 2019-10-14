import numpy as np
import copy

import gym
import gridworld



class A2CAgent(object):
	"""docstring for A2CAgent"""
	def __init__(self, action_space, gamma, ):
		super(A2CAgent, self).__init__()
		self.gamma = gamma

		self.pi = # NN
		self.V = # NN

		# agir et recevoir un s' et un r
		




	def act(self, observation, reward, done):
		pass


if __name__ == '__main__':


    env = gym.make('CartPole-v1')

    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 1000000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()