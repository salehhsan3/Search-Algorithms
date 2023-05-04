import numpy as np

from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict
import queue

class GraphNode:
    def __init__(self,state,parent,action,cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.in_OPEN = False
        self.in_CLOSE = False



class BFSAgent():
    def __init__(self) -> None:
        self.OPEN = queue.SimpleQueue()  #FIFO QUEUE
        self.CLOSE = set()
        self.expanded_counter = 0.0

    def get_path(self,node: GraphNode) -> Tuple[List[int],int,float]:
        pass
########################################################################################3
    def exist_in_queue(self,q,state):
        #######################33
       pass


    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        initial_node = GraphNode(env.s,None,None,None)
        initial_node.in_OPEN = True
        self.OPEN.put(initial_node)

        while not self.OPEN.empty():
            node_to_expand = self.OPEN.get()
            self.expanded_counter += 1
            self.CLOSE.add(node_to_expand.state)
            node_to_expand.in_CLOSE = True
            node_to_expand.in_OPEN = False

            for act,tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                if terminated:
                    if env.is_final_state(next_state):
                        final_node = GraphNode(next_state, node_to_expand, act, cost)
                        return self.get_path(final_node)
                if self.exist_in_queue(self.OPEN,next_state) or next_state in self.CLOSE:
                    pass
                else:
                    new_node = GraphNode(next_state,node_to_expand,act,cost)
                    self.OPEN.put(new_node)









class DFSAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError
        


class UCSAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError



class GreedyAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError

class WeightedAStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], int, float]:
        raise NotImplementedError   


class IDAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError