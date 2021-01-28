import os
import pickle
import string
import time
import logging
import numpy as np
import importlib
import sys


import wandb


from absl import flags
from absl import logging
from absl import app

""" Greedy and Beam Search"""
from StandardHC import N2Greedy_invM as Greedy
from StandardHC import beamSearchOptimal_invM as beam_search
from StandardHC import likelihood_invM as likelihood

""" Exact Trellis"""
from ClusterTrellis.run_experiments import compare_map_gt_and_bs_trees as compute_trellis
from ClusterTrellis import HierarchicalTrellis
from ClusterTrellis.trellis_node import TrellisNode
from ClusterTrellis.Ginkgo_node import ModelNode

""" A star trellis """
from AstarTrellis.iter_trellis3 import IterJetTrellis
from AstarTrellis.iter_trellis_maxSteps import IterJetTrellis as ApproxIterJetTrellis

"""Replace with model auxiliary scripts to calculate the energy function"""
# from ClusterTrellis import Ginkgo_likelihood as likelihood


# from ClusterTrellis.utils import get_logger
# logger = logging.get_logger(level=logging.WARNING)

NleavesMin=4
tcut=2.5
Ntrees = 5
powerset = 2**NleavesMin

# flags.DEFINE_string('a_star_trellis_class', 'IterJetTrellis', 'Type of Algorithm')
# flags.DEFINE_string('trellis_class', 'Approx_IterJetTrellis', 'Type of Algorithm')

flags.DEFINE_string("wandb_dir", "/Users/sebastianmacaluso/Documents/HCmanager", "wandb directory - If running seewp process, run it from there")
# flags.DEFINE_string('dataset_dir', "../../data/Ginkgo/input/", "dataset dir ")
flags.DEFINE_string('dataset_dir', "../../../ginkgo/data/invMassGinkgo/", "dataset dir ")
# flags.DEFINE_string('dataset', "test_" + str(NleavesMin) + "_jets.pkl", 'dataset filename')
flags.DEFINE_string('dataset', "jets_"+str(NleavesMin)+"N_"+str(Ntrees)+"trees_"+str(int(10*tcut))+"tcut_.pkl", 'dataset filename')

flags.DEFINE_string('output_dir', "../../data/Ginkgo/output/", "output dir ")
flags.DEFINE_string('results_filename', "out_jets_"+str(NleavesMin)+"N_"+str(Ntrees)+"trees_"+str(int(10*tcut))+"tcut_.pkl", 'results filename')

flags.DEFINE_integer('max_steps', 4000, 'Maximum number of steps')
flags.DEFINE_integer('max_nodes', powerset + 10, 'nodes')
flags.DEFINE_string('exp_name', 'AStar', 'name')
flags.DEFINE_string('output', 'exp_out', 'output directory')
flags.DEFINE_integer('num_points', 12, '')
flags.DEFINE_integer('seed', 42, '')
flags.DEFINE_string('child_func', 'all_two_partitions', 'function used to get children when initializing nodes')
flags.DEFINE_integer('num_repeated_map_values', 0, 'number of times the same MAP value is returned before halting') #This was for the approx. A*. Not implemented for Ginkgo. Set to 0.
flags.DEFINE_integer('propagate_values_up', 0, 'whether to propagate f,g,h values during trellis extension.')
flags.DEFINE_integer("beam_size", 3*NleavesMin, "Beam size") #Beam Search


FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)



class GinkgoEvaluator:
    def __init__(self, redraw_existing_jets=False):
        # self.filename = filename
        # self.env = env
        self.redraw_existing_jets =redraw_existing_jets
        self.methods = []  # Method names
        self.log_likelihoods = {}  # Log likelihood results
        self.illegal_actions = {}  # Number of illegal actions
        self.likelihood_evaluations = {}
        self.times ={}
        self.tree_size = []

        # if os.path.exists(filename) and not redraw_existing_jets:
        self.trees = self._load()[0:3]
        logging.info("# Trees = %s", len(self.trees))
        logging.info("========"*5)
        # else:
        #     self.n_jets = n_jets
        #     self.jets = self._init_jets()
        #     self._save()

    def eval_true(self, method):
        self.tree_size = [ len(jet["leaves"]) for jet in self.trees]
        log_likelihoods = [self._compute_true_log_likelihood(jet) for jet in self.trees]
        illegal_actions = [0 for _ in self.trees]
        likelihood_evaluations = [0 for _ in self.trees]
        times = [0 for _ in self.trees]
        self._update_results(method, log_likelihoods, illegal_actions, times, likelihood_evaluations)
        logging.info(" Truth values = %s", log_likelihoods)
        # return log_likelihoods, illegal_actions, likelihood_evaluations

    def eval_greedy(self, method):
        temp = np.asarray([self._compute_greedy_log_likelihood(jet) for jet in self.trees])
        temp = temp.transpose()
        log_likelihoods = temp[0]
        times = temp[1]
        illegal_actions = [0 for _ in self.trees]
        likelihood_evaluations = [
            self._compute_greedy_likelihood_evaluations(jet) for jet in self.trees
        ]
        self._update_results(method, log_likelihoods, illegal_actions, times, likelihood_evaluations)
        logging.info(" Greedy values = %s", log_likelihoods)
        # return log_likelihoods, illegal_actions, likelihood_evaluations

    def eval_beam_search(self, method, beam_size=4):
        temp = np.asarray([self._compute_beam_search_log_likelihood(jet) for jet in self.trees])
        temp = temp.transpose()
        log_likelihoods = temp[0]
        times = temp[1]
        illegal_actions = [0 for _ in self.trees]
        likelihood_evaluations = [
            self._compute_beam_search_likelihood_evaluations(jet, beam_size) for jet in self.trees
        ]
        self._update_results(method, log_likelihoods, illegal_actions, times, likelihood_evaluations)
        logging.info(" Beam Search values = %s", log_likelihoods)
        # return log_likelihoods, illegal_actions, likelihood_evaluations

    def eval_exact_trellis(self, method):
        temp = np.asarray([self._compute_exact_trellis_log_likelihood(jet) for jet in self.trees])
        temp = temp.transpose()
        log_likelihoods = temp[0]
        times = temp[1]
        illegal_actions = [0 for _ in self.trees]
        likelihood_evaluations = [0 for _ in self.trees]
        self._update_results(method, log_likelihoods, illegal_actions, times, likelihood_evaluations)
        logging.info(" Exact trellis MAP values = %s",log_likelihoods )
        # return log_likelihoods, illegal_actions, likelihood_evaluations


    def eval_exact_a_star(self, method):
        temp = np.asarray([self._compute_a_star_log_likelihood(jet) for jet in self.trees])
        temp = temp.transpose()
        log_likelihoods = temp[0]
        times = temp[1]
        logging.info("log LH = %s", log_likelihoods)
        logging.info("times = %s", times)
        logging.info("+++++++"*10)
        # log_likelihoods = [self._compute_a_star_log_likelihood(jet)[0] for jet in self.trees]
        # times = [self._compute_a_star_log_likelihood(jet)[1] for jet in self.trees]
        illegal_actions = [0 for _ in self.trees]
        likelihood_evaluations = [0 for _ in self.trees]
        self._update_results(method, log_likelihoods, illegal_actions, times, likelihood_evaluations)
        logging.info(" A star MAP values = %s",log_likelihoods )
        # return log_likelihoods, illegal_actions, likelihood_evaluations

    def eval_approx_a_star(self, method):
        temp = np.asarray([self._compute_approx_a_star_log_likelihood(jet) for jet in self.trees])
        temp = temp.transpose()
        log_likelihoods = temp[0]
        times = temp[1]
        illegal_actions = [0 for _ in self.trees]
        likelihood_evaluations = [0 for _ in self.trees]
        self._update_results(method, log_likelihoods, illegal_actions, times, likelihood_evaluations)
        logging.info(" Approx. star MAP values = %s",log_likelihoods )
        # return log_likelihoods, illegal_actions, likelihood_evaluations


    ###------
    def _update_results(self, method, log_likelihoods, illegal_actions, times, likelihood_evaluations):
        self.log_likelihoods[method] = log_likelihoods
        self.illegal_actions[method] = illegal_actions
        self.times[method] = times
        self.likelihood_evaluations[method] = likelihood_evaluations

    # def _save(self):
    #     data = {"n_jets": self.n_jets, "jets": self.jets}
    #     with open(self.filename, "wb") as file:
    #         pickle.dump(data, file)

    def _load(self):
        indir = FLAGS.dataset_dir
        in_filename = os.path.join(indir, FLAGS.dataset)
        if os.path.exists(in_filename) and not self.redraw_existing_jets:
            with open(in_filename, "rb") as fd:
                data = pickle.load(fd, encoding='latin-1')
        else:
            logging.info("Please choose a dataset")
            data = []
        return data

    def save(self):
        out_filename = os.path.join(FLAGS.output_dir, FLAGS.results_filename)
        if os.path.exists(FLAGS.output_dir):
            with open(out_filename, "wb") as f:
                pickle.dump((self.tree_size,self.log_likelihoods, self.illegal_actions,self.times, self.likelihood_evaluations ), f, protocol=2)



    @staticmethod
    def _compute_true_log_likelihood(jet):
        return sum(jet["logLH"])


    @staticmethod
    def _compute_greedy_log_likelihood(jet):
        startTime = time.time()
        greedy_jet = Greedy.recluster(
            jet,
            delta_min=jet["pt_cut"],
            lam=float(jet["Lambda"]),
            visualize=True)

        Time = time.time() - startTime

        return [sum(greedy_jet["logLH"]), Time]


    @staticmethod
    def _compute_beam_search_log_likelihood(jet, beam_size = 5):

        startTime = time.time()
        n = len(jet["leaves"])
        bs_jet = beam_search.recluster(
            jet,
            # beamSize=min(beam_size, n * (n - 1) // 2),
            beamSize= beam_size,
            delta_min=jet["pt_cut"],
            lam=float(jet["Lambda"]),
            N_best=1,
            visualize=False,
        )[0]

        Time = time.time() - startTime

        return [sum(bs_jet["logLH"]), Time]


    @staticmethod
    def _compute_exact_trellis_log_likelihood(tree, max_leaves=11):

        if len(tree["leaves"]) > max_leaves:
            return np.nan


        data_params = tree
        N = len(data_params['leaves'])

        """Replace with current model parameters"""
        leaves_features = [[data_params['leaves'][i], 0] for i in range(N)]
        model_params = {}
        model_params["delta_min"] = float(data_params['pt_cut'])
        model_params["lam"] = float(data_params['Lambda'])

        startTime = time.time()
        """ Create and fill the trellis"""
        trellis, Z, map_energy, Ntrees, totTime = compute_trellis(tree,
                                                                  ModelNode,
                                                                  leaves_features,
                                                                  model_params)

        Time = time.time() - startTime

        return [map_energy, Time]


    @staticmethod
    def _compute_a_star_log_likelihood(tree, max_leaves=11):

        if len(tree["leaves"]) > max_leaves:
            return np.nan

        gt_jet = tree
        startTime = time.time()

        trellis = IterJetTrellis(leaves=gt_jet['leaves'],
                                 propagate_values_up=FLAGS.propagate_values_up,
                                 max_nodes=FLAGS.max_nodes,
                                 min_invM=gt_jet['pt_cut'],
                                 Lambda=gt_jet['Lambda'],
                                 LambdaRoot=gt_jet['LambdaRoot'])

        hc, MAP_value, step = trellis.execute_search(num_matches=FLAGS.num_repeated_map_values,
                                                     max_steps=int(FLAGS.max_steps))

        Time = time.time() - startTime
        logging.info(f'total time = {Time}')
        logging.info(f'FINAL f ={MAP_value}')
        logging.info("-------------------------------------------")
        logging.info("=====++++++" * 5)

        return [MAP_value, Time]


    @staticmethod
    def _compute_approx_a_star_log_likelihood(tree, max_leaves=11):

        if len(tree["leaves"]) > max_leaves:
            return np.nan

        gt_jet = tree
        startTime = time.time()

        trellis = ApproxIterJetTrellis(leaves=gt_jet['leaves'],
                                       propagate_values_up=FLAGS.propagate_values_up,
                                       max_nodes=FLAGS.max_nodes,
                                       min_invM=gt_jet['pt_cut'],
                                       Lambda=gt_jet['Lambda'],
                                       LambdaRoot=gt_jet['LambdaRoot'])

        hc, MAP_value, step = trellis.execute_search(num_matches=FLAGS.num_repeated_map_values,
                                                     max_steps=int(FLAGS.max_steps))


        Time = time.time() - startTime
        logging.info(f'total time = {Time}')
        logging.info(f'FINAL f ={MAP_value}')
        logging.info("-------------------------------------------")

        return [MAP_value, Time]


    #-------------
    @staticmethod
    def _compute_greedy_likelihood_evaluations(jet):

        n = len(jet["leaves"])
        # beam_size = min(beam_size, n * (n - 1) // 2),
        evaluations = 0

        while n > 1:
            evaluations += n * (n - 1) // 2
            n -= 1

        return evaluations


    @staticmethod
    def _compute_beam_search_likelihood_evaluations(jet, beam_size =5):

        n = len(jet["leaves"])
        # beam_size = min(beam_size, n * (n - 1) // 2),
        beam = 1
        evaluations = 0

        while n > 1:
            evaluations += beam * n * (n - 1) // 2
            beam = beam_size
            n -= 1

        return evaluations

def main(argv):

    wandb.init(project="%s" % (FLAGS.exp_name), dir=FLAGS.wandb_dir)
    wandb.config.update(flags.FLAGS)
    np.random.seed(FLAGS.seed)

    Evaluator = GinkgoEvaluator()
    Evaluator.eval_true("truth")
    Evaluator.eval_greedy("Greedy")
    Evaluator.eval_beam_search("BS", beam_size = FLAGS.beam_size)
    Evaluator.eval_exact_trellis("exact_trellis")
    Evaluator.eval_exact_a_star("exact_a_star")
    Evaluator.eval_approx_a_star("approx_a_star")

    Evaluator.save()

    logging.info("Tree size =%s",Evaluator.tree_size)
    logging.info("log_likelihoods= %s", Evaluator.log_likelihoods)
    logging.info("Illegal actions =  %s" , Evaluator.illegal_actions)
    logging.info("Times = %s ",   Evaluator.times)
    logging.info("Number of evaluations =  %s", Evaluator.likelihood_evaluations)

if __name__ == '__main__':
    app.run(main)

# sys.exit()