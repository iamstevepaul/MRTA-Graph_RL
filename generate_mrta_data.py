import argparse
import os
import numpy as np
from utils.data_utils import save_dataset


def generate_mrta_data(dataset_size, mrta_size):

    return list(zip(
        np.random.uniform(size=(dataset_size, 1, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, mrta_size, 2)).tolist(),  # Node locations
        np.random.uniform(0.1, 1,  size=(dataset_size, mrta_size)).tolist() ,  # deadlines
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, default='mrta', help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='mrta',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=100, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[50, 100, 200, 500, 1000],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    problem = "mrta"



    for graph_size in opts.graph_sizes:

        datadir = os.path.join(opts.data_dir, problem)
        os.makedirs(datadir, exist_ok=True)


        np.random.seed(opts.seed)

        dataset = generate_mrta_data(opts.dataset_size, graph_size)

        save_dataset(dataset, datadir+"/"+graph_size+"_nodes_+"+problem+".pkl")
