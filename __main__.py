import sys
from algorithms.W_PathSim import W_PathSim


def main():
    input_file = './dataset/dblp/dblp_small.gexf'
    metapath = 'Author-Paper-Author'
    # metapath = 'Author-Paper-Venue-Paper-Author'
    model = W_PathSim(input_file, metapath)
    source_node_id = '19926'  # aJiaweiHan
    k = 10  # get top-10 results
    top_10_sim_nodes = model.get_top_k_similar_nodes(source_node_id, k)
    print(
        'Top-{} similar nodes of: [{}] node:'.format(k, model.C_HIN.nodes[source_node_id][model.C_HIN.NODE_NAME_ATTR]))
    for (node_id, sim_score) in top_10_sim_nodes:
        print('{}:\t{}'.format(model.C_HIN.nodes[node_id][model.C_HIN.NODE_NAME_ATTR], sim_score))


if __name__ == '__main__':
    sys.exit(main())
