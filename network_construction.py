import networkx as nx
import multiprocessing
from rdkit import DataStructs, Chem
import utils
import copy
import time
import skbio
from skbio import Protein
from skbio.alignment._pairwise import blosum50
from tqdm import tqdm
#from skbio.alignment import local_pairwise_align_ssw
#skbio.alignment.local_pairwise_align_protein
#MTTSASSHLNKGIKQVYMSLPQGEKVQAMYIWIDGTGEGLRCKTRTLDSEPKCVEELPEWNFDGSSTLQSEGSNSDMYLVPAAMFRDPFRKDPNKLVLCEVFKYNRRPAETNLRHTCKRIMDMVSNQHPWFGMEQEYTLMGTDGHPFGWPSNGFPGPQGPYYCGVGADRAYGRDIVEAHYRACLYAGVKIAGTNAEVMPAQWEFQIGPCEGISMGDHLWVARFILHRVCEDFGVIATFDPKPIPGNWNGAGCHTNFSTKAMREENGLKYIEEAIEKLSKRHQYHIRAYDPKGGLDNARRLTGFHETSNINDFSAGVANRSASIRIPRTVGQEKKGYFEDRRPSANCDPFSVTEALIRTCLLNETGDEPFQYKN
def create_target_similarity_network(target_seq, name):
    target_names = list(target_seq.keys())
    seq_info = []
    total = len(target_names)
    for i in tqdm(range(len(target_names))):
        seq1 = target_seq[target_names[i]]
        #print(seq1)
        #raise Exception('stop')
        for j in range(i+1, len(target_names)):
            #print(i,'  ',j)
            seq2 = target_seqs[target_names[j]]
            try:
                alignment, score, start_end_positions = skbio.alignment.local_pairwise_align_ssw(Protein(seq1), Protein(seq2), substitution_matrix = blosum50)
            except:
                score = 0
            #alignment, score, start_end_positions = skbio.alignment.local_pairwise_align_protein(Protein(p1_s), Protein(p2_s))
            seq_info.append([target_names[i], target_names[j], score])
        #t2 = time.time()
        #print(t2-t1)
    #print(seq_info[0:10])
    name = 'data/'+name+'target_similarity.pkl'
    utils.save_any_obj_pkl(seq_info, name)

def create_target_similarity_network_normalised(target_seq, name):
    import math
    target_names = list(target_seq.keys())
    seq_info = []
    total = len(target_names)
    for i in tqdm(range(len(target_names))):
        seq1 = target_seq[target_names[i]]
        #print(seq1)
        #raise Exception('stop')
        for j in range(i+1, len(target_names)):
            #print(i,'  ',j)
            seq2 = target_seqs[target_names[j]]
            try:
                alignment, score, start_end_positions = skbio.alignment.local_pairwise_align_ssw(Protein(seq1), Protein(seq2), substitution_matrix = blosum50)
            except:
                score = 0
            #alignment, score, start_end_positions = skbio.alignment.local_pairwise_align_protein(Protein(p1_s), Protein(p2_s))
            new_score = float(score)/(math.sqrt(len(seq1))*math.sqrt(len(seq2)))
            seq_info.append([target_names[i], target_names[j], new_score])
        #t2 = time.time()
        #print(t2-t1)
    #print(seq_info[0:10])
    name = 'data/'+name+'target_similarity.pkl'
    utils.save_any_obj_pkl(seq_info, name)

"""
def create_target_similarity_network(target_seq, name, top_ratio = 0.04):
    target_names = list(target_seq.keys())
    seq_info = []
    total = len(target_names)

    for i in range(len(target_names)):
        print(i)
        seq1 = target_seq[target_names[i]]
        p1 = target_seqs[target_names[0]].split('\n')[1:-1]
        p1_s = ''
        for item in p1:
            p1_s += item
        for j in range(i+1, len(target_names)):
            #print(i,'  ',j)
            p2 = target_seqs[target_names[1]].split('\n')[1:-1]
            p2_s = ''
            for item in p2:
                p2_s += item
            alignment, score, start_end_positions = skbio.alignment.local_pairwise_align_ssw(Protein(p1_s), Protein(p2_s), substitution_matrix = blosum50)
            #alignment, score, start_end_positions = skbio.alignment.local_pairwise_align_protein(Protein(p1_s), Protein(p2_s))
            seq_info.append([target_names[i], target_names[j], score])
        #t2 = time.time()
        #print(t2-t1)
    #print(seq_info[0:10])
    name = 'data/'+name+'target_similarity_hs.pkl'
    utils.save_any_obj_pkl(seq_info, name)
"""
def sw_similarity():
    csd = utils.load_any_obj_pkl('data/chem_seq_dict.pkl')
    chem_id = list(csd.keys())
    #for item in chem_id:
    #    if 
        #if len(csd[item]) < 5 :
        #    print('get it')
    print(chem_id[0])
    print(chem_id[1])
    match = 2
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring)  # you can also choose gap penalties, etc...
    alignment = sw.align(csd[chem_id[0]],csd[chem_id[1]])
    #score = alignment.dump()
    print(alignment.score)

def get_edges_from_matrix(matrix):
    h_names = []
    h_number = len(matrix)
    for i in range(h_number):
        h_names.append('h_'+str(i))
    c_names = []
    c_number = len(matrix[0])
    for i in range(c_number):
        c_names.append('c_'+str(i))
    edges = []
    for i_index in range(h_number):
        for j_index in range(c_number):
            if matrix[i_index][j_index] == '1':
                edges.append([h_names[i_index], c_names[j_index]])
    print('h_number:',h_number)
    print('c number:',c_number)
    print('total number of edges:',len(edges))
    return edges, h_names, c_names
    
def construct_network_from_edges(edges):
    G = nx.Graph()
    for item in edges:
        G.add_edge(item[0], item[1], weight = 1)
    return G

def add_all_nodes(graph, all_nodes):
    print('add all nodes')
    new_graph = graph.copy()
    for node in all_nodes:
        #if node == 'h_0':
        #    print("found you")
        new_graph.add_node(node)
    #print(new_graph['h_0'])
    return new_graph

def create_compound_similarity_network_mp(compounds_smiles_path, species_name = '_DB', worker = 4, top_ratio = 0.04):
    compounds_smiles = utils.load_any_obj_pkl(compounds_smiles_path)
    all_compounds = list(compounds_smiles.keys())
    #print(Chem.SanitizeMol('CN(CCO[P@](O)(=O)O[P@@](O)(=O)O[Be-](F)(F)F)C1=CC=CC=C1[N+]([O-])=O'))
    #for item in all_compounds:
    #    m2 = Chem.MolFromSmiles(compounds_smiles[item])
    #    if m2 == None:
    #        print(item)
    #        print(compounds_smiles[item])
    #raise Exception('stop')
    ccd = calculate_molecular_similarity(compounds_smiles, worker = worker)
    all_corr = ccd.parallel_calculate_all_correlation()
    #all_corr = [[str(j) for j in i] for i in all_corr]
    final_corr = []
    for item in all_corr:
        for a_corr in item:
            #print(a_corr)
            final_corr.append(a_corr)
    save_name = 'data/'+'compound_similarity'+species_name+'.pkl'
    utils.save_any_obj_pkl(final_corr, save_name)

def get_most_significant_edge(all_corrs, top_ratio = 0.04):
    import pandas as pd
    df = pd.DataFrame(all_corrs)#, columns=['target1','target2','value']
    n_number = int(len(all_corrs)* 0.04)
    interested_edges = df.nlargest(n_number, 2)
    edges = interested_edges.values.tolist()
    #print(edges)
    return edges

def construct_signifcant_edge_network(all_corrs, top_ratio = 0.04):
    all_nodes = []
    for a_edge in all_corrs:
        all_nodes.append(a_edge[0])
        all_nodes.append(a_edge[1])
    all_nodes = list(set(all_nodes))
    sim_network = nx.Graph()
    for node in all_nodes:
        sim_network.add_node(node)
    import pandas as pd
    df = pd.DataFrame(all_corrs)#, columns=['target1','target2','value']
    n_number = int(len(all_corrs) * top_ratio)
    interested_edges = df.nlargest(n_number, 2)
    edges = interested_edges.values.tolist()
    #print(edges)
    for edge in edges:
        sim_network.add_edge(edge[0], edge[1], weight = edge[2])
    return sim_network

def create_implicit_networks(drug_target_network, interested_nodes):
    G = nx.Graph()
    #for node in interested_nodes:
    #    G.add_node(node)
    for i in range(len(interested_nodes)):
        #print(i,' out of ',len(interested_nodes))
        G.add_node(interested_nodes[i])
        for j in range(i+1, len(interested_nodes)):
            if i == j:
                continue
            neighbours_i = set(drug_target_network[interested_nodes[i]])
            neighbours_j = set(drug_target_network[interested_nodes[j]])
            common = list(neighbours_i.intersection(neighbours_j))
            edge_weight = 0
            new_edge_weight = 0
            for item in common:
                i_node_weight = drug_target_network[interested_nodes[i]][item]['weight']
                j_node_weight = drug_target_network[interested_nodes[j]][item]['weight']
                #print(i_node_weight,'   ',j_node_weight)
                #if i_node_weight != 1 or j_node_weight != 1:
                #    print('wtf')
                new_weight = i_node_weight*j_node_weight
                new_edge_weight+=new_weight
            if new_edge_weight != 0:
                G.add_edge(interested_nodes[i],interested_nodes[j],weight = new_edge_weight)
    return G
                
def network_cc(edges, compounds_smiles):
    compounds_smiles = []
    G = nx.Graph()
    for item in edges:
        G.add_edge(item[0], item[1], weight= 1)
    cc = sorted(nx.connected_components(G), key = len, reverse=True)
    max_size = 0
    max_cc_compounds = []
    for cc_item in cc:
        if len(cc_item)> max_size:
            max_size = len(cc_item)
            max_cc_compounds = cc_item
    return max_cc_compounds

def create_compound_similarity_network(compounds_smiles, top_ratio = 0.04):
    compounds_name = list(compounds_smiles.keys())
    similarity_info = []
    total = len(compounds_name)
    for i in range(len(compounds_name)):
        #t1 = time.time()
        #print(compounds_smiles[compounds_name[i]])
        #print(i,' out of ',total)
        m1 = Chem.MolFromSmiles(compounds_smiles[compounds_name[i]])
        fps1 = Chem.RDKFingerprint(m1)
        for j in range(i+1, len(compounds_name)):
            m2 = Chem.MolFromSmiles(compounds_smiles[compounds_name[i]])
            fps2 = Chem.RDKFingerprint(m2)
            simialrity_coefficient = DataStructs.FingerprintSimilarity(fps1,fps2)
            similarity_info.append([compounds_name[i], compounds_name[j], simialrity_coefficient])
        #t2 = time.time()
        #print(t2-t1)
    utils.save_any_obj_pkl('data/all_interactions.pkl')

class calculate_molecular_similarity:
    def __init__(self, compounds_smiles, worker = 4): # recommend: parallel_walks = number_walks; if memory error, plz reduce it
        self.compounds = list(compounds_smiles.keys())
        self.compounds_smiles_dict = compounds_smiles
        self.worker = worker
        self.length = len(compounds_smiles.keys())

    def parallel_calculate_all_correlation(self):
        #t1 = time.time()
        print('start parallel computationing')
        print('worker number:',self.worker)
        pool = multiprocessing.Pool(processes=self.worker)
        range_number = len(self.compounds)-1
        all_similarities = pool.map(self.calculate_similarity, range(range_number))
        pool.close()  # Waiting for all subprocesses done..
        pool.join()
        return all_similarities

    def calculate_similarity(self, compound_index):
        print(compound_index)
        similarity_info = []
        m1 = Chem.MolFromSmiles(self.compounds_smiles_dict[self.compounds[compound_index]], sanitize = False)
        fps1 = Chem.RDKFingerprint(m1)
        for i in range(compound_index+1, self.length):
            m2 = Chem.MolFromSmiles(self.compounds_smiles_dict[self.compounds[i]], sanitize = False)
            fps2 = Chem.RDKFingerprint(m2)
            simialrity_coefficient = DataStructs.FingerprintSimilarity(fps1,fps2)
            similarity_info.append([self.compounds[compound_index], self.compounds[i], simialrity_coefficient])
        return similarity_info

        
if __name__ == "__main__":
    #create_compound_similarity_network_mp('data/drugbank_drugs.pkl', species_name = '_DB')#60 
    #sw_similarity()#cp sa mm
    #Chem.SanitizeMol
    
    
    #all_keys = list(target_seqs.keys())
    #a = target_seqs[all_keys[0]]
    #print(type(a))
    #target_IDs = list(target_seqs.keys())
    #print(target_seqs[target_IDs[0]])
    #t1 = target_IDs[0]
    #seqs1 = target_seqs[t1].split('\n')
    #for item in target_IDs:
    #    seqs1 = target_seqs[item]
    #    print(len(seqs1.split('\n')[-3]))

    #create_compound_similarity_network_mp('data/drugbank_drugs.pkl', species_name = '_DB')
    target_seqs = utils.load_any_obj_pkl('data/drugbank_targets.pkl')
    create_target_similarity_network(target_seqs, 'DB_N_')
    
    #alignment, score, start_end_positions = skbio.alignment.local_pairwise_align_protein(Protein(p1_s), Protein(p2_s))
    #print(score)
