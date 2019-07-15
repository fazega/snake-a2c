import numpy as np
import variables
import time
# import tensorflow as tf


from agent_process import AgentProcess
from multiprocessing import Process, Pipe
import threading

class MasterProcess():
    def __init__(self, verbose=False):
        self.processes = {}

    def train_agents(self):
        pipes = {}
        for i in range(0, variables.n_process+1):
            parent_conn, child_conn = Pipe()
            pipes[i] = parent_conn
            p = AgentProcess(conn=child_conn, id=i, n_games=variables.n_per_process)
            p.start()
            self.processes[i] = p

        scores = {}
        batchs = {}
        t0 = time.time()
        def listenToAgent(id, scores):
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

        window = 50000 // (variables.n_process*variables.n_per_process)
        iter = 1
        mean_scores = []
        file = open("log_scores", "w")
        while True:
            if(len(scores) == variables.n_process):
                id_best = min(scores, key=scores.get)
                mean_scores.append(np.mean(list(scores.values())))
                print("End of iteration "+str(iter)+". Mean score sor far : "+str(np.mean(mean_scores)))
                iter += 1
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
