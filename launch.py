"""Master process, spawning subprocesses to act and learn."""

from multiprocessing import Process, Pipe
import itertools
import numpy as np
import time
import threading


from agent_process import AgentProcess
import variables

WINDOW_SCORES = 50_000


def launch() -> None:
    """Launches an experiment."""
    processes = {}
    pipes = {}
    # We add a +1 for the first process.
    for i in range(variables.n_process + 1):
        parent_connection, child_connection = Pipe()
        pipes[i] = parent_connection
        agent_process = AgentProcess(
            conn=child_connection, id=i, n_games=variables.n_per_process)
        agent_process.start()
        processes[i] = agent_process

    scores = {}
    batchs = {}
    t0 = time.time()
    def listenToAgent(id: int, scores: dict[int, float]):
        while True:
            msg = pipes[id].recv()
            if msg == "saved":
                print("Master process (0) saved his weights.")
                for j in pipes:
                    if(j != 0):
                        pipes[j].send("load")
            else:
                score = float(msg[0])
                scores[id] = score
                batchs[id] = msg[1]
                print("Process "+str(id)+" returns score "+str(score))

    threads_listen = []
    print("Threads to start")
    for id in pipes:
        t = threading.Thread(target=listenToAgent, args=(id,scores))
        t.start()
        threads_listen.append(t)
    print("Threads started")

    window = WINDOW_SCORES // (variables.n_process*variables.n_per_process)
    mean_scores = []
    file = open("log_scores", "w")
    step = 0
    while True:
        if(len(scores) == variables.n_process):
            id_best = min(scores, key=scores.get)
            mean_scores.append(np.mean(list(scores.values())))
            print("End of iteration "+str(step)+". Mean score sor far : "+str(np.mean(mean_scores)))
            step += 1
            file.write(str(np.mean(mean_scores))+"\n")
            file.flush()
            print("Time : "+str(time.time()-t0))
            print("\n")
            pipes[0].send(("train_with_batchs", list(batchs.values())))
            t0 = time.time()
            scores.clear()
            batchs.clear()

        if(len(mean_scores) >= window):
            mean_scores = mean_scores[1:]

if __name__ == "__main__":
    launch()
