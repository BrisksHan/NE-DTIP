'''
Static Network Embedding: DeepWalk
'''

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import time
import random
import pickle

import gensim
import logging
import numpy as np
import networkx as nx
#import note2vec_sampling
import utils
from gensim.models.poincare import PoincareModel
from random import shuffle
import copy
from itertools import chain
import math
import multiprocessing

class EpochLogger():
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


class DeepWalk(object):
     def __init__(self, G_dynamic, emb_dim=5, num_walks=30, walk_length=80, window=10, 
                    workers=1, negative=10, seed=42, restart_prob=None, walks_workers = 1):#negative 5
          self.G_dynamic = G_dynamic.copy()  # a series of dynamic graphs
          self.emb_dim = emb_dim   # node emb dimensionarity
          self.restart_prob = restart_prob  # restart probability for random walks; if None --> DeepWalk
          self.num_walks = num_walks  # num of walks start from each node
          self.walk_length = walk_length  # walk length for each walk
          self.window = window  # Skip-Gram parameter
          self.workers = workers  # Skip-Gram parameter
          self.negative = negative  # Skip-Gram parameter
          self.seed = seed  # Skip-Gram parameter
          self.walks_workers = walks_workers
          self.emb_dicts = [] # emb_dict @ t0, t1, ...; len(self.emb_dicts) == len(self.G_dynamic)
          
     def sampling_traning(self):     
          for t in range(len(self.G_dynamic)):
               t1 = time.time()
               G0 = self.G_dynamic[t]
               epoch_logger = EpochLogger()
               print('walk worker:', self.walks_workers,'  SGNS worker:',self.workers)
               w2v = gensim.models.Word2Vec(sentences=None, size=self.emb_dim, window=self.window, sg=1, hs=0, negative=self.negative, ns_exponent=0.75,
                                   alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4, workers=self.workers, seed=self.seed,
                                   corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                                   max_vocab_size=None, max_final_vocab=None, trim_rule=None)  # w2v constructor, default parameters
               if self.walks_workers <= 1:
                    print('start random walks with only 1 core')
                    sentences = simulate_walks(nx_graph=G0, num_walks=self.num_walks, walk_length=self.walk_length, restart_prob=None) #restart_prob=None or 0 --> deepwalk
                    sentences = [[str(j) for j in i] for i in sentences]
               else:
                    print('start random walks with multiple cores')
                    sw_obj = mp_walker(nx_graph=G0, num_walks = self.num_walks, walk_length = self.walk_length, worker = self.walks_workers)
                    sentences = sw_obj.simulate_walks()
                    sentences = [[str(j) for j in i] for i in sentences]
               
               
               w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
               w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor

               emb_dict = {} # {nodeID: emb_vector, ...}
               for node in self.G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]
               self.emb_dicts.append(emb_dict)
               t2 = time.time()
               print(f'DeepWalk sampling and traning time: {(t2-t1):.2f}s --> {t+1}/{len(self.G_dynamic)} graphs')
          
          return self.emb_dicts  # To save memory useage, we can delete DynWalks model after training

     def save_emb(self, path='unnamed_dyn_emb_dicts.pkl'):
          ''' save # emb_dict @ t0, t1, ... to a file using pickle
          '''
          with open(path, 'wb') as f:
               pickle.dump(self.emb_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
     
     def load_emb(self, path='unnamed_dyn_emb_dicts.pkl'):
          ''' load # emb_dict @ t0, t1, ... to a file using pickle
          '''
          with open(path, 'rb') as f:
               any_object = pickle.load(f)
          return any_object




# ----------------------------------------------------------------------------------------------------
# ---------- utils: most_affected_nodes, simulate_walks, random_walk, random_walk_restart ------------
# ----------------------------------------------------------------------------------------------------

#def node2vec_walks(nx_graph, num_walks, walk_length, restart_prob=None, affected_nodes=None, p = 1, q = 2):
#     node2vec_graph = note2vec_sampling.Graph(graph=nx_graph, is_directed=False, p=p, q=q, alpha= 0, feature_sim = None, seed = 2014)
#     node2vec_graph.preprocess_transition_probs()
#     walks = node2vec_graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)
#     return walks
    
def sliding_windows(walks, window_size = 10, reverse_pair = True):
     pairs = []
     for i in range(len(walks)):
          for j in range(len(walks[i])):
               new_pairs = get_neighbor_pairs(walks[i], j, window_size, reverse_pair = reverse_pair)
               pairs += new_pairs
     print(pairs[0:100])
     shuffle(pairs)
     #print(pairs[0]," ",pairs[1])
     #print(pairs[0]," ",pairs[1])
     print(pairs[0:100])
     return pairs

def get_neighbor_pairs(single_walk, current_index, window_size, reverse_pair = True):
     start_index = current_index - window_size
     end_index = current_index + window_size
     result = []
     for i in range(start_index, end_index):
          if start_index != current_index and i > 0 and i < len(single_walk):
               current_pair = (single_walk[current_index], single_walk[i])               
               result.append(current_pair)
               if reverse_pair == True:
                    current_pair = (single_walk[i], single_walk[current_index])
                    result.append(current_pair)
     return result

#del poincare_ball_model(relations, nodes, dimension = 2, workers = 4):
#     #wv = model.kv.word_vec('kangaroo.n.01')
#     print("training poincare_disk_model")
#     model = PoincareModel(relations, negative=10, size=dimension, negative=20, workers=4)
#     model.train(epochs=50)
#     return model

def poincare_disk_model(relations, dimension = 2, workers = 1, negative_sample = 2, batch_number = 10):
     #for i in range(100):
     #     print(relations[i])
     print("poincare ball model initialization")
     model = PoincareModel(relations, negative = negative_sample, size = dimension, workers = workers)
     print("start poincare ball model training")
     #batch = int(len(relations)/batch_number)
     #print("batch size: ",batch)
     model.train(epochs = 50, print_every=1000, batch_size = 100000)
     return model

def get_top_k_neighbors(nx_graph, k = 10):
     print('start searching top neighbors')
     node_list = list(nx_graph.nodes())
     node_close_neighbors = {}
     for i in range(len(node_list)):
          current_node_edge_weight_list = []
          for j in range(len(node_list)):
               if i == j:
                    current_node_edge_weight_list.append(0)
               else:
                    weight = nx_graph[node_list[i]][node_list[j]]['weight']
                    current_node_edge_weight_list.append(weight)
          indices = np.argpartition(current_node_edge_weight_list, -k)[-k:]
          top_neighbor_IDs = []
          for j in range(len(indices)):
               top_neighbor_IDs.append(node_list[indices[j]])
          node_close_neighbors[node_list[i]] = top_neighbor_IDs
     print('finished searching top neighbors')
     return node_close_neighbors

def simulate_walks(nx_graph, num_walks, walk_length, restart_prob=None, affected_nodes=None):
     '''
     Repeatedly simulate random walks from each node
     '''
     G = nx_graph
     walks = []
     print('start random walks')
     if affected_nodes == None: # simulate walks on every node in the graph [offline] --> deepwalk
          nodes = list(G.nodes())
     else:                     # simulate walks on affected nodes [online]
          nodes = list(affected_nodes)
     
     if restart_prob == None: # naive random walk
          t1 = time.time()
          print('start time:',t1)
          i = 0
          for walk_iter in range(num_walks):
               print('iteration:',i)
               i+=1
               random.shuffle(nodes)
               for node in nodes:
                    walks.append(random_walk(nx_graph=G, start_node=node, walk_length=walk_length))
          t2 = time.time()
          print(f'random walk sampling, time cost: {(t2-t1):.2f}')
     else: # random walk with restart
          t1 = time.time()
          for walk_iter in range(num_walks):
               random.shuffle(nodes)
               for node in nodes:
                    walks.append(random_walk_restart(nx_graph=G, start_node=node, walk_length=walk_length, restart_prob=restart_prob))
          t2 = time.time()
          print(f'random walk sampling, time cost: {(t2-t1):.2f}')
     return walks

def randomly_select_weighted_neighbors(nx_graph, start_node):
     #import random
     #random.choices(['one', 'two', 'three'], [0.2, 0.3, 0.5], k=10)
     cur_nbrs = list(nx_graph.neighbors(start_node))
     cur_nhrs_weight = []
     for i in range(len(cur_nbrs)):
          weight = nx_graph[cur_nbrs[i]][start_node]['weight']
          cur_nhrs_weight.append(weight)
     #total_weights = sum(cur_nhrs_weight)
     #for i in range(len(cur_nhrs_weight)):
     #     cur_nhrs_weight[i] = cur_nhrs_weight[i]/total_weights
     selected_nhr = random.choices(cur_nbrs, weights=cur_nhrs_weight)
     #print(selected_nhr[0])
     return selected_nhr[0]

def random_walk(nx_graph, start_node, walk_length):
    '''
    Simulate a random walk starting from start node
    '''
    G = nx_graph
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            cur = randomly_select_weighted_neighbors(nx_graph, cur)
            walk.append(cur)
        else:
            break
    return walk


def random_walk_restart(nx_graph, start_node, walk_length, restart_prob):
    '''
    random walk with restart
    restart if p < restart_prob
    '''
    G = nx_graph
    walk = [start_node]

    while len(walk) < walk_length:
        p = random.uniform(0, 1)
        if p < restart_prob:
            cur = walk[0] # restart
            walk.append(cur)
        else:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
    return walk


class mp_walker:
     #nx_graph, num_walks, walk_length, restart_prob=None, affected_nodes=None
     def __init__(self, nx_graph, num_walks, walk_length, worker = 3, restart_prob=None, affected_nodes=None):
          self.nx_graph = nx_graph
          self.num_walks = num_walks
          self.walk_length = walk_length
          self.worker = worker
          print('walks worker number', self.worker)
          #self.path = 'current_correlation_graph.pkl'
          #utils.save_any_obj_pkl(nx_graph, self.path)

     def simulate_walks(self):
          print('star simultating walks')
          t1 = time.time()
          pool = multiprocessing.Pool(processes=self.worker)
          all_walks = pool.map(self.walks_wp, range(self.num_walks))
          pool.close()  # Waiting for all subprocesses done..
          pool.join()
          #print(len(list(self.nx_graph.nodes)))
          #print(len(all_walks))
          all_walks = list(chain(*all_walks))
          #print(len(all_walks))
          t2 = time.time()
          print(f'Time for all random walks: {(t2-t1):.2f}')  # use multiple cores, total time < sum(time@itr)
          return all_walks

     def walks_wp(self, iteration):
          print('start a iteration', iteration)
          walks = []
          random.seed()                             # *** for multiprocessor version
          np.random.seed()                          # *** do NOT remove these 'random' operation
          nodes = list(self.nx_graph.nodes)    # *** otherwise, each number_walks may give the same node sequences...
          random.shuffle(nodes)                     # *** which hence decrease performance
          t1 = time.time()
          for node in nodes:
               walks.append(self.deepwalk_walks(start_node=node))
          t2 = time.time()
          print(f'Walk iteration: {iteration+1}/{self.num_walks}; time cost: {(t2-t1):.2f}')
          return walks

     def deepwalk_walks(self, start_node):
          #simulate_walks(nx_graph, num_walks, walk_length, restart_prob=None, affected_nodes=None)
          walk = [start_node]
          local_length = self.walk_length
          while len(walk) < local_length:
               cur = walk[-1]
               cur_nbrs = list(self.nx_graph.neighbors(cur))
               if len(cur_nbrs) > 0:
                    new_cur = self.randomly_select_weighted_neighbors_mp(cur)
                    walk.append(new_cur)
               else:
                    break
          return walk

     def randomly_select_weighted_neighbors_mp(self, start_node):
          nx_graph = self.nx_graph
          cur_nbrs = list(nx_graph.neighbors(start_node))
          cur_nbrs_weight = []
          for i in range(len(cur_nbrs)):
               weight = nx_graph[cur_nbrs[i]][start_node]['weight']
               cur_nbrs_weight.append(weight)
          #total_weights = sum(cur_nhrs_weight)
          #for i in range(len(cur_nhrs_weight)):
          #     cur_nhrs_weight[i] = cur_nhrs_weight[i]/total_weights
          if len(cur_nbrs) > 0:
               selected_nhr = random.choices(cur_nbrs, weights=cur_nbrs_weight)
               return selected_nhr[0]#added [0] duot o return a node
          else:
               raise Exception('node should not be isolated as this is the second node')


if __name__ == "__main__":
     #relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat'), ('a', 'b')]
     #poincare_disk_model(relations)
     #nx_graph = utils.load_any_obj_pkl('245000and')
     #graph = gene_correlation_graph.create_soft_threshold_graph(new_sample_names, new_cluster_expressions, power)
     #print(nx_graph[0])
     #get_top_k_neighbors(nx_graph)
     arr = [10,9,8,7,11,12,13,1,2,3]
     n = 3
     indices = np.argpartition(arr, -n)[-n:]
     print(indices)