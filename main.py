import gymnasium
import matplotlib.pyplot as plt
from gym import wrappers

# https://pypi.org/project/colorama/
from colorama import just_fix_windows_console, init, Back, Fore

init(autoreset=True)

# use Colorama to make Termcolor work on Windows too
just_fix_windows_console()


env = gymnasium.make("ALE/DoubleDunk-v5", render_mode = "human")



ambiente, info = env.reset(seed=42)



# # print(ambiente_txt, "Tamanho: " + str(len(ambiente)))

while True:
  # definição da política
  env.render()
  acao = env.action_space.sample()

  ambiente, recompensa, finalizado, paralizado, info = env.step(acao)
  
  recompensa_txt = "Recompensa: [%s]" % recompensa
  fim_txt = "Finalizado: [%s]" % finalizado
  acao_txt = "Ação: [%s]" % acao

  if (recompensa < 0):
    recompensa_txt = Fore.RED + recompensa_txt + Fore.RESET
  elif (recompensa >= 0):
    recompensa_txt = Fore.GREEN + recompensa_txt + Fore.RESET

  if (finalizado):
    fim_txt = Fore.RED + fim_txt + Fore.RESET

  print(recompensa_txt, fim_txt, acao_txt)

  if finalizado or paralizado:
    break
    #ambiente, info = env.reset()
env.close()