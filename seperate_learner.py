import DeepWalk_edge_weight
import numpy as np
import utils
#sklearn 0.23.2

#sklearn version on bluebear 0.21.3


class two_stage_learning():
    def __init__(self, DTI_network, compound_list, target_list, tsn_network, worker, csn_network, implicit_t_network, implicit_c_network, wl, nn, wn, embedding_files = None, load_emb = True, dim = 128, normalise_emb = False):
        self.DTI_network = DTI_network
        self.tsn_network = tsn_network
        self.csn_network = csn_network
        self.implicit_t_network = implicit_t_network
        self.implicit_c_network = implicit_c_network
        self.compound_list = compound_list
        self.target_list= target_list
        self.worker = worker
        self.load_emb = load_emb
        self.embedding_files = embedding_files
        self.dim = dim
        self.normalise_emb = normalise_emb
        self.wn = wn
        self.nn = nn
        self.wl = wl

    def learn_all_network_embedding(self):
        def learn_embedding(network):
            DW = DeepWalk_edge_weight.DeepWalk([network], workers = self.worker, emb_dim = self.dim , num_walks=self.wn, walk_length=self.wl, window=10, walks_workers= self.worker, negative=self.nn)#now win = 10
            tsn_emb = DW.sampling_traning()[0]
            return tsn_emb
        
        print('train emb without consider reload')
        print('learn tsn network embdding')
        self.tsn_embedding = learn_embedding(self.tsn_network)
        print('learn csn network embdding')
        self.csn_embedding = learn_embedding(self.csn_network)
        print('learn ti network embdding')
        self.ti_embedding = learn_embedding(self.implicit_t_network)
        print('learn ci network embdding')
        self.ci_embedding = learn_embedding(self.implicit_c_network)
    
    def build_node_representation(self):
        print("start to build node representation")
        target_embs = {}
        targets = self.target_list

        def normalise_emb(emb_dict):
            new_dict = {}
            keys = list(emb_dict.keys())
            for item in keys:
                current_emb = emb_dict[item]
                max_num = max(current_emb)
                normalized = current_emb/max_num
                new_dict[item] = normalized
            return new_dict

        if self.normalise_emb == True:
            self.tsn_embedding = normalise_emb(self.tsn_embedding)
            self.ti_embedding = normalise_emb(self.ti_embedding)
            self.csn_embedding = normalise_emb(self.csn_embedding)
            self.ci_embedding = normalise_emb(self.ci_embedding)


        for target in targets:
            #if target == 'hsa768':
            #    print("to be completetd")
            target_tsn = self.tsn_embedding[target]
            target_ti = self.ti_embedding[target]
            target_embs[target] = np.concatenate((target_tsn, target_ti), axis=None)
            #print(len(target_tsn),'  ',len(target_ti),'   ',len(np.concatenate((target_tsn, target_ti), axis=None)))
            #raise Exception('stop')
        compound_emb = {}
        compounds = self.compound_list
        for compound in compounds:
            compound_csn = self.csn_embedding[compound]
            compound_ci = self.ci_embedding[compound]
            compound_emb[compound] = np.concatenate((compound_csn, compound_ci), axis=None)
        self.target_embs = target_embs
        self.compound_embs = compound_emb

    def train_DTI_prediction_svm(self, training_samples, training_labels, kernal = 1):
        def pairwise_kernel(vector1, vector2):
            gram_matrix = np.zeros((len(vector1), len(vector2)))#, dtype="float16"
            dim = int(len(vector1[0])/2)
            from tqdm import tqdm
            for vector1_index in tqdm(range(len(vector1))):
                #if vector1_index/1 == 0:
                #print('kernel:',vector1_index,'   ',len(vector1))
                for vector2_index in range(len(vector2)):
                    #value_sum = 0
                    #for i in range(dim):
                    #    for j in range(dim):
                    #        value_sum += vector1[vector1_index][i]*vector2[vector2_index][i]*vector1[vector1_index][j+dim]*vector2[vector2_index][j+dim]
                    #gram_matrix[vector1_index, vector2_index]  = value_sum
                    v11 = np.array(vector1[vector1_index][0:dim])
                    v12 = np.array(vector1[vector1_index][dim:])
                    v21 = np.array(vector2[vector2_index][0:dim])
                    v22 = np.array(vector2[vector2_index][dim:])
                    gram_matrix[vector1_index, vector2_index] = np.dot(v11,v21) * np.dot(v12,v22)

        if kernal == 0:
            selected_kernel = 'linear'
        elif kernal == 1:
            selected_kernel = 'poly'
        elif kernal == 2:
            selected_kernel = 'rbf'
        elif kernal == 3:
            selected_kernel = 'sigmoid'
        elif kernal == 4:
            selected_kernel = pairwise_kernel
        else:    
            raise Exception('wrong choice, please input 0, 1, 2, 3, 4')
        from sklearn import svm
        print('linear start training with kernel ',selected_kernel)
        clf = svm.SVC(probability=True, kernel = selected_kernel, verbose=False)
        clf.fit(training_samples, training_labels)
        return clf
    
    def construct_training_samples(self, negative_ratio = 10):
        print('concate samples embedding')
        postive_edges = list(self.DTI_network.edges())
        training_samples = []
        training_labels = []
        target_list = self.target_list
        compound_list = self.compound_list
        sorted_postive_edges = []
        for edge in postive_edges:
            node1 = edge[0]
            node2 = edge[1]
            #print(node1,'   ',node2)
            if node1 in compound_list:
                current_sample = list(self.compound_embs[node1]) + list(self.target_embs[node2])
                training_samples.append(current_sample)
                training_labels.append(1)
                sorted_postive_edges.append([node1,node2])
            else:
                current_sample = list(self.compound_embs[node2]) + list(self.target_embs[node1])
                training_samples.append(current_sample)
                training_labels.append(1)
                sorted_postive_edges.append([node2,node1])
        negtive_num = len(postive_edges)*negative_ratio
        print('nagative ratio:',negative_ratio)
        print('negative num:',negtive_num)
        pair_num = 0
        negtive_num_max = len(target_list) * len(compound_list) - len(sorted_postive_edges)
        if negtive_num > negtive_num_max:
            negtive_num = negtive_num_max
        while pair_num < negtive_num:
            if pair_num %10000 == 0:
                print(pair_num," out of ",negtive_num)
            c = np.random.choice(compound_list)
            t = np.random.choice(target_list)
            non_edge = [c, t]
            if [c, t] not in sorted_postive_edges:
                pair_num += 1
                #training_samples.append(non_edge)
                #training_labels.append(0)
                node1 = non_edge[0]
                node2 = non_edge[1]
                current_sample = list(self.compound_embs[node1]) + list(self.target_embs[node2])
                training_samples.append(current_sample)
                training_labels.append(0)

        return training_samples, training_labels

    def construct_all_unknown_pairs(self, negative_ratio):
        print('concate all samples embedding')
        postive_edges = list(self.DTI_network.edges())
        print('DTI edges num:',len(postive_edges))
        training_samples = []
        training_labels = []
        target_list = self.target_list
        compound_list = self.compound_list
        for edge in postive_edges:
            node1 = edge[0]
            node2 = edge[1]
            #print(node1,'   ',node2)
            if node1 in compound_list:
                current_sample = list(self.compound_embs[node1]) + list(self.target_embs[node2])
                training_samples.append(current_sample)
                training_labels.append(1)
            else:
                current_sample = list(self.compound_embs[node2]) + list(self.target_embs[node1])
                training_samples.append(current_sample)
                training_labels.append(1)
        negtive_num = len(postive_edges)*negative_ratio
        print('nagative ratio:',negative_ratio)
        print('negative num:',negtive_num)
        pair_num = 0
        from tqdm import tqdm
        for compound in tqdm(compound_list):
            for target in target_list:
                non_edge = [compound, target]
                if ([compound, target] not in postive_edges) and ([target,compound] not in postive_edges):
                    pair_num += 1
                    #training_samples.append(non_edge)
                    #training_labels.append(0)
                    node1 = non_edge[0]
                    node2 = non_edge[1]
                    current_sample = list(self.compound_embs[node1]) + list(self.target_embs[node2])
                    training_samples.append(current_sample)
                    training_labels.append(0)

        return training_samples, training_labels

    def concatenate_pair_embeddings(self, node_pairs):
        compound_list = self.compound_list
        samples = []
        for pair in node_pairs:
            node1 = pair[0]
            node2 = pair[1]
            if node1 in compound_list:
                current_sample = list(self.compound_embs[node1]) + list(self.target_embs[node2])
                samples.append(current_sample)
            else:
                current_sample = list(self.compound_embs[node2]) + list(self.target_embs[node1])
                samples.append(current_sample)
        return samples


    
            