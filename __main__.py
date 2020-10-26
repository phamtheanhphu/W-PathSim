import json
import sys

from algorithms.W_PathSim import W_PathSim
from evaluation_metrics.nDCG_evaluator import nDCG_Evaluator


def find_topk_venues(input_file):
    sun_venue_data_file_path = './dataset/dblp/sun_dataset_venues.json'
    venue_topic_data_file_path = './dataset/dblp/gs_venue_topics.json'

    venue_json_data = json.load(open(sun_venue_data_file_path, 'r', encoding='utf-8'))
    venue_topic_json_data = json.load(open(venue_topic_data_file_path, 'r', encoding='utf-8'))

    # metapath = 'Venue-Paper-Venue'
    metapath = 'Venue-Paper-Author-Paper-Venue'
    model = W_PathSim(input_file, metapath)

    source_node_id = '2504'  # vKDD
    source_node_name = model.C_HIN.nodes[source_node_id][model.C_HIN.NODE_NAME_ATTR]
    k = 10  # get top-10 results
    top_10_sim_nodes = model.get_top_k_similar_nodes(source_node_id, k)
    print('Top-{} similar nodes of: [{}] node:'.format(k, source_node_name))

    results = []
    for (node_id, sim_score) in top_10_sim_nodes:
        print(' - {}:\t{}'.format(model.C_HIN.nodes[node_id][model.C_HIN.NODE_NAME_ATTR], sim_score))
        results.append(model.C_HIN.nodes[node_id][model.C_HIN.NODE_NAME_ATTR])

    # Evaluating the outputs by using the nDCG metric
    nDCG_evaluator = nDCG_Evaluator(venue_json_data, venue_topic_json_data)
    nDCG_score = nDCG_evaluator.calculate_nDCG_score(source_node_name, results)
    print('The nDCG accuracy score is: [{}]'.format(nDCG_score))


def find_topk_authors(input_file):
    sun_author_data_file_path = './dataset/dblp/sun_dataset_authors.json'
    author_topic_data_file_path = './dataset/dblp/acm_author_topics.json'

    author_json_data = json.load(open(sun_author_data_file_path, 'r', encoding='utf-8'))
    author_topic_json_data = json.load(open(author_topic_data_file_path, 'r', encoding='utf-8'))

    # metapath = 'Author-Paper-Author'
    metapath = 'Author-Paper-Venue-Paper-Author'
    model = W_PathSim(input_file, metapath)

    source_node_id = '19926'  # aJiaweiHan
    source_node_name = model.C_HIN.nodes[source_node_id][model.C_HIN.NODE_NAME_ATTR]
    k = 10  # get top-10 results
    top_10_sim_nodes = model.get_top_k_similar_nodes(source_node_id, k)
    print('Top-{} similar nodes of: [{}] node:'.format(k, source_node_name))

    results = []
    for (node_id, sim_score) in top_10_sim_nodes:
        print(' - {}:\t{}'.format(model.C_HIN.nodes[node_id][model.C_HIN.NODE_NAME_ATTR], sim_score))
        results.append(model.C_HIN.nodes[node_id][model.C_HIN.NODE_NAME_ATTR])

    # Evaluating the outputs by using the nDCG metric
    nDCG_evaluator = nDCG_Evaluator(author_json_data, author_topic_json_data)
    nDCG_score = nDCG_evaluator.calculate_nDCG_score(source_node_name, results)
    print('The nDCG accuracy score is: [{}]'.format(nDCG_score))


def main():
    input_file = './dataset/dblp/dblp_small.gexf'
    # find_topk_authors(input_file)
    print('\n---\n')
    find_topk_venues(input_file)


if __name__ == '__main__':
    sys.exit(main())
