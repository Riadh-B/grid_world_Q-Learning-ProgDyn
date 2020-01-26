import sys
from agent import Agent

if __name__ == "__main__":
    epochs_count = 50
    if (len(sys.argv) > 1):
        epochs_count = int(sys.argv[1])
    ag = Agent()
    print("initial Q-values ... \n")
    for key in ag.Q_values:
        print(str(key), '->', str(ag.Q_values[key]))

    ag.play(epochs_count)
    print("latest Q-values ... \n")
    for key in ag.Q_values:
        print(str(key), '->', str(ag.Q_values[key]))