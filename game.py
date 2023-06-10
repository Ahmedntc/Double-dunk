import gym
from colorama import just_fix_windows_console, init, Back, Fore
import numpy as np
import collections
import pickle

init(autoreset=True)
try:
    pickle_file = open('q_table.pickle', 'rb')
    q_table = pickle.load(pickle_file)
    pickle_file.close()
except:
    print("Arquivo não encontrado, criando novo arquivo")
    q_table = collections.defaultdict(float)


# use Colorama to make Termcolor work on Windows too
just_fix_windows_console()

env = gym.make("ALE/DoubleDunk-v5", render_mode=None, obs_type='ram')
ambiente, info = env.reset(seed=42)

# Q-learning

q_table = collections.defaultdict(float)

learning_rate = 0.1
discount_factor = 0.9

epocas = 27_000 # aproximadamente 7 horas no meu pc

def get_max_q(s):
    max_q = -99999
    for action in range(env.action_space.n):
        estado_acao = (s, action)
        if q_table[estado_acao] > max_q:
            max_q = q_table[estado_acao]

    return max_q

for epoca in range(epocas):
    s_state = tuple(env.reset()[0])
    print("Epoca: ", epoca)
    while True:

        rand_action = np.random.random()

        if rand_action < 0.1:
            action = env.action_space.sample()
        else:
            action = np.argmax([q_table[(s_state, a)] for a in range(env.action_space.n)])

        
        step = env.step(action)
        n_state = tuple(step[0])
        reward = step[1]
        # print("Action: ", action)
        # print("n_state: " , n_state)

        estado_acao = (s_state, action)

        q_table[estado_acao] = q_table[estado_acao] + (learning_rate *(reward + discount_factor * (get_max_q(s_state)) - q_table[estado_acao]))

        s_state = n_state[0]

        if step[2]:
            break

        if step[3]:
            print("Falha!")
            break

    # output the file every 1000 epocas
    if epoca % 1000 == 0:
        pickle_file = open('q_table.pickle', 'wb')
        pickle.dump(q_table, pickle_file)
        pickle_file.close()

# output the file
pickle_file = open('q_table.pickle', 'wb')
pickle.dump(q_table, pickle_file)
pickle_file.close()

env.close()