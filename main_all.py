from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import utils
import network_construction
#import DeepWalk_edge_weight
import seperate_learner
import normalise_sample_representation
import numpy as np

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    # -----------------------------------------------general settings--------------------------------------------------
    #parser.add_argument('--dti_path', default='data/NE-IN_DTI.pkl')
    #parser.add_argument('--drug_sim_path', default = 'data/NE-IN_drug_similarity.pkl')
    #parser.add_argument('--target_sim_path',default = 'data/NE-IN_target_similarity.pkl')
    parser.add_argument('--dti_path', default='data/ic_DTI.pkl')
    parser.add_argument('--drug_sim_path', default = 'data/ic_drug_similarity.pkl')
    parser.add_argument('--target_sim_path',default = 'data/ic_target_similarity.pkl')
    parser.add_argument('--worker', default=4)
    parser.add_argument('--dim',default = 128)
    parser.add_argument('--ns', default=0)#strategy 0 , 1, 2
    parser.add_argument('--walk_num', default=10)
    parser.add_argument('--walk_length', default=80)
    parser.add_argument('--negative_number', default=5)
    parser.add_argument('--sparsity', default=0.04)
    parser.add_argument('--output_name', default='unkown')
    args = parser.parse_args()
    return args


def run_two_stage(args):
    DTI_network = utils.load_any_obj_pkl(args.dti_path)
    drug_similarity = utils.load_any_obj_pkl(args.drug_sim_path)
    target_similarity = utils.load_any_obj_pkl(args.target_sim_path)

    csn_network = network_construction.construct_signifcant_edge_network(drug_similarity, top_ratio=float(args.sparsity))
    tsn_network = network_construction.construct_signifcant_edge_network(target_similarity, top_ratio=float(args.sparsity))

    implicit_compounds = network_construction.create_implicit_networks(DTI_network, list(csn_network.nodes()))
    implicit_targets = network_construction.create_implicit_networks(DTI_network, list(tsn_network.nodes()))

    learner = seperate_learner.two_stage_learning(DTI_network = DTI_network, compound_list = list(csn_network.nodes()), target_list = list(tsn_network.nodes()),tsn_network = tsn_network, csn_network = csn_network, implicit_t_network = implicit_targets, implicit_c_network = implicit_compounds, wl=int(args.walk_length), nn=int(args.negative_number), wn = int(args.walk_num), worker = int(args.worker), load_emb=False)
    learner.learn_all_network_embedding()
    learner.build_node_representation()
    
    training_samples, training_labels = learner.construct_training_samples(negative_ratio= 10)

    test_pairs = new_pairs_to_evaludate(list(csn_network.nodes()), list(tsn_network.nodes()), DTI_network)
    test_samples = learner.concatenate_pair_embeddings(test_pairs)

    training_samples = normalise_sample_representation.standardscaler_transform(training_samples)
    test_samples = normalise_sample_representation.standardscaler_transform(test_samples)

    clf = learner.train_DTI_prediction_svm(training_samples, training_labels, kernal = 2)
    probs = clf.predict_proba(test_samples)
    new_probs = [row[1] for row in probs]
    all_evaluation = []
    #from tqdm import tqdm
    for i in range(len(test_pairs)):
        current_one = [test_pairs[i][0], test_pairs[i][1], new_probs[i]]
        all_evaluation.append(current_one)
    output_name = 'output/'+args.output_name+'.pkl'
    utils.save_any_obj_pkl(all_evaluation, output_name)

def new_pairs_to_evaludate(drug_list, target_list, DTI):
    pairs = []
    for a_drug in drug_list:
        current_drug_neighbours = list(DTI[a_drug])
        non_edges = list(set(target_list)-set(current_drug_neighbours))
        for a_target in non_edges:
            pairs.append([a_drug, a_target])
    return pairs

if __name__ == "__main__":
    run_two_stage(parse_args())