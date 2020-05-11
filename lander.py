import gym
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy import reshape, mean
from utils.deep_q_network import DeepQNetwork


env = gym.make('LunarLander-v2')
env.seed(0)
seed(0)


def train_network(epochs):
    loss = []
    agent = DeepQNetwork(env.observation_space.shape[0], env.action_space.n)
    for i in range(epochs):
        state = env.reset()
        state = reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        for _ in range(max_steps):
            action = agent.get_optimal_action(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = reshape(next_state, (1, 8))
            agent.update_memory(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(i, epochs, score))
                break
        loss.append(score)
        if i % 100 == 0:
            agent.save_memory()
        # Average score of last 100 episode
        is_solved = mean(loss[-100:])
        if is_solved > 200:
            print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss


if __name__ == '__main__':
    print(env.observation_space)
    print(env.action_space)
    episodes = 500
    training_loss = train_network(episodes)
    plt.plot([i + 1 for i in range(0, len(training_loss), 2)], training_loss[::2])
    plt.show()
