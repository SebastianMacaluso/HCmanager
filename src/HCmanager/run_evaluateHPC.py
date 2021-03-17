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
from AstarTrellis.iter_trellis_exact import IterJetTrellis
# from AstarTrellis.iter_trellis_approx import IterJetTrellis as ApproxIterJetTrellis
from AstarTrellis.iter_trellis_approx_fullbeam import IterJetTrellis as ApproxIterJetTrellis

"""Replace with model auxiliary scripts to calculate the energy function"""
# from ClusterTrellis import Ginkgo_likelihood as likelihood


# from ClusterTrellis.utils import get_logger
# logger = logging.get_logger(level=logging.WARNING)

NleavesMin=15
tcut=2.5
Ntrees = 1
# powerset = 2**NleavesMin


HPC=True
if HPC:
    flags.DEFINE_integer('NleavesMin', None, 'Number of elements of the trees datasets')
    flags.DEFINE_string('dataset_dir', "../../../ginkgo/data/invMassGinkgo/", "dataset dir ")
    # flags.DEFINE_string('dataset',
    #                     "jets_" + str(NleavesMin) + "N_" + str(Ntrees) + "trees_" + str(int(10 * tcut)) + "tcut_.pkl",
    #                     'dataset filename')
    flags.DEFINE_string('dataset', "jets_6N_10trees_25tcut_0.pkl", 'dataset filename')
    flags.DEFINE_string("wandb_dir", "/scratch/sm4511/HCmanager", "wandb directory - If running seewp process, run it from there")
    # flags.DEFINE_string('output_dir', "../../data/Ginkgo/output/", "output dir ")
    # flags.DEFINE_string('results_filename', "out_jets_" + str(NleavesMin) + "N_" + str(Ntrees) + "trees_" + str(
    #     int(10 * tcut)) + "tcut_.pkl", 'results filename')
    flags.DEFINE_string('algorithm', None, "Algorithm to run the scan")

else:
    flags.DEFINE_integer('NleavesMin', NleavesMin, 'Number of elements of the trees datasets')
    # flags.DEFINE_string('dataset', "jets_80N_5trees_1tcut_0.pkl", 'dataset filename')
    flags.DEFINE_string('dataset', "test_" + str(NleavesMin) + "_jets.pkl", 'dataset filename')
    # flags.DEFINE_string("wandb_dir", "/Users/sebastianmacaluso/Documents/HCmanager",
    #                     "wandb directory - If running seewp process, run it from there")
    # flags.DEFINE_integer('NleavesMin', None, 'Number of elements of the trees datasets')
    flags.DEFINE_string('dataset_dir', "../../data/Ginkgo/input/", "dataset dir ")

    # flags.DEFINE_string('dataset_dir', "../../../ginkgo/data/invMassGinkgo/", "dataset dir ")
    # flags.DEFINE_string('dataset', "jets_"+str(NleavesMin)+"N_"+str(Ntrees)+"trees_"+str(int(10*tcut))+"tcut_.pkl", 'dataset filename')
    flags.DEFINE_string("wandb_dir", "../..",
                        "wandb directory - If running seewp process, run it from there")

    flags.DEFINE_string('algorithm', "ApproxAstar", "Algorithm to run the scan")


flags.mark_flag_as_required('NleavesMin')
# flags.DEFINE_integer('id', 0, 'job id (to run on HPC')

# flags.DEFINE_string('a_star_trellis_class', 'IterJetTrellis', 'Type of Algorithm')
# flags.DEFINE_string('trellis_class', 'Approx_IterJetTrellis', 'Type of Algorithm')


flags.mark_flag_as_required('algorithm')




flags.DEFINE_string('output_dir', "../../data/Ginkgo/output/", "output dir ")
flags.DEFINE_string('results_filename', "out_jets_"+str(NleavesMin)+"N_"+str(Ntrees)+"trees_"+str(int(10*tcut))+"tcut_.pkl", 'results filename')

flags.DEFINE_integer('max_steps', 5000000000, 'Maximum number of steps (Only relevant for the aprox. Astar algorithm')
# flags.DEFINE_integer('max_nodes', powerset + 10, 'nodes')
flags.DEFINE_string('exp_name', 'AStar', 'name')
flags.DEFINE_string('output', 'exp_out', 'output directory')
# flags.DEFINE_integer('num_points', 12, '')
flags.DEFINE_integer('seed', 42, '')
flags.DEFINE_string('child_func', 'all_two_partitions', 'function used to get children when initializing nodes')
flags.DEFINE_integer('num_repeated_map_values', 0, 'number of times the same MAP value is returned before halting') #This was for the approx. A*. Not implemented for Ginkgo. Set to 0.
flags.DEFINE_integer('propagate_values_up', 0, 'whether to propagate f,g,h values during trellis extension.')
# flags.DEFINE_integer("beam_size", 3*NleavesMin, "Beam size") #Beam Search

flags.DEFINE_integer('max_leaves', 20, 'Maximum number of leaves to run exact trellis and Astar algorithms. If above, return NaN')
<<<<<<< Updated upstream
flags.DEFINE_integer('all_pairs_max_size', 10, 'Maximum number of elements of a node to run the exact algorithm - switch to approx. algo for more elements')
# flags.DEFINE_multi_integer('num_tries', [5,3], 'List with [Number of samples to draw, best pair (based on likelihood) to keep]')
flags.DEFINE_multi_integer('num_tries_m', 6, 'Number of samples to draw')
flags.DEFINE_multi_integer('num_tries_k', 3, 'Best pair (based on likelihood) to keep')

flags.DEFINE_integer('maxBeamJets', 10000000, 'Maximum number of leaves to run exact trellis and Astar algorithms. If above, return NaN')
# flags.DEFINE_integer('maxBeamJets', 1, 'Maximum number of leaves to run exact trellis and Astar algorithms. If above, return NaN')
flags.DEFINE_integer('max_nodes', None , 'Maximum number of nodes for the sparse trellis')
flags.mark_flag_as_required('max_nodes')


flags.DEFINE_boolean("exact_heuristic_proof", False, "If true run algorithm with exact heuristic that has a proof")
flags.DEFINE_boolean("approx_heuristic", False, "If true run algorithm with approximate heuristic, if False then run it with a supposedly exact heuristic checked on plots with up to 9 elements but with no proof")

flags.DEFINE_boolean("exact_astar_approx_heuristic", False, "If true run exact a star using approx heuristics")
=======
flags.DEFINE_integer('all_pairs_max_size', 7, 'Maximum number of elements of a node to run the exact algorithm - switch to approx. algo for more elements')
flags.DEFINE_multi_integer('num_tries', [8,5], 'List with [Number of samples to draw, best pair (based on likelihood) to keep]')

flags.DEFINE_integer('maxBeamJets', 10000000, 'Maximum number of leaves to run exact trellis and Astar algorithms. If above, return NaN')
# flags.DEFINE_integer('maxBeamJets', 1, 'Maximum number of leaves to run exact trellis and Astar algorithms. If above, return NaN')

>>>>>>> Stashed changes

FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)



class GinkgoEvaluator:
    def __init__(self, num_tries=None, redraw_existing_jets=False):
        # self.filename = filename
        # self.env = env
        self.num_tries=num_tries
        self.redraw_existing_jets =redraw_existing_jets
        self.methods = []  # Method names
        self.log_likelihoods = {}  # Log likelihood results
        self.illegal_actions = {}  # Number of illegal actions
        self.likelihood_evaluations = {}
        self.times ={}
        self.tree_size = []
        self.NodesExplored = {}
        self.empty1 = {}
        # self.empty2 = {}
        self.Ntrees = {}
        self.BStrees=[]

        # if os.path.exists(filename) and not redraw_existing_jets:
        self.trees = self._load()[0:1]
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

    def eval_beam_search(self, method, beam_size=4, MLEtree=False):
        if MLEtree:
            temp = np.asarray([self._compute_beam_search_log_likelihood(jet, beam_size=beam_size, MLEtree=MLEtree) for jet in self.trees])
            self.BStrees=temp
        else:
            temp = np.asarray([self._compute_beam_search_log_likelihood(jet, beam_size=beam_size) for jet in self.trees])
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

    def eval_exact_trellis(self, method, max_leaves=11):
        temp = np.asarray([self._compute_exact_trellis_log_likelihood(jet, max_leaves=max_leaves) for jet in self.trees])
        temp = temp.transpose()
        log_likelihoods = temp[0]
        times = temp[1]
        self.empty1 = temp[2] #Partition function values
        illegal_actions = [0 for _ in self.trees]
        likelihood_evaluations = [0 for _ in self.trees]
        self._update_results(method, log_likelihoods, illegal_actions, times, likelihood_evaluations)
        logging.info(" Exact trellis MAP values = %s",log_likelihoods )
        # return log_likelihoods, illegal_actions, likelihood_evaluations


    def eval_exact_a_star(self, method, max_leaves=11,  max_nodes = None):
        temp = np.asarray([self._compute_a_star_log_likelihood(jet, max_leaves=max_leaves, max_nodes = max_nodes) for jet in self.trees])
        temp = temp.transpose()
        log_likelihoods = temp[0]
        times = temp[1]
        steps = temp[2]
        self.NodesExplored[method] = temp[3]
        self.Ntrees[method]  = temp[4]
        logging.info("log LH = %s", log_likelihoods)
        logging.info("times = %s", times)
        logging.info("+++++++"*10)
        # log_likelihoods = [self._compute_a_star_log_likelihood(jet)[0] for jet in self.trees]
        # times = [self._compute_a_star_log_likelihood(jet)[1] for jet in self.trees]
        illegal_actions = [0 for _ in self.trees]
        # likelihood_evaluations = [0 for _ in self.trees]
        self._update_results(method, log_likelihoods, illegal_actions, times, steps)
        logging.info(" A star MAP values = %s",log_likelihoods )
        # return log_likelihoods, illegal_actions, likelihood_evaluations

    def eval_approx_a_star(self, method, beam_size = 5, exact_heuristic_proof = False, approx_heuristic=False, max_nodes = None, info=None):
        temp = np.asarray([self._compute_approx_a_star_log_likelihood(jet,beam_size=beam_size, exact_heuristic_proof = exact_heuristic_proof, approx_heuristic = approx_heuristic, max_nodes = max_nodes) for jet in self.trees])
        temp = temp.transpose()
        log_likelihoods = temp[0]
        times = temp[1]
        steps = temp[2]
        self.NodesExplored[method] = temp[3]
        self.Ntrees[method]  = temp[4]
        self.empty1 = info
        illegal_actions = [0 for _ in self.trees]
        # likelihood_evaluations = [0 for _ in self.trees]
        self._update_results(method, log_likelihoods, illegal_actions, times, steps)
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

    def save(self, output_dir, ApproxAstar=False):
        if ApproxAstar:
            run_details = "_MaxSize_"+str(FLAGS.all_pairs_max_size)+"_NumTries_"+str(self.num_tries[0])+"_"+str(self.num_tries[1])
            filename = FLAGS.results_filename.strip().split(".")[0] + run_details + "." + FLAGS.results_filename .strip().split(".")[1]
        else:
            filename = FLAGS.results_filename

        out_file = os.path.join(output_dir,filename)
        if os.path.exists(output_dir):
            with open(out_file, "wb") as f:
                pickle.dump((self.tree_size,self.log_likelihoods, self.illegal_actions,self.times, self.likelihood_evaluations, self.NodesExplored, self.empty1, self.Ntrees ), f, protocol=2)



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
<<<<<<< Updated upstream
    def _compute_beam_search_log_likelihood(jet, beam_size = 5,N_best=1, FullTree=False, MLEtree=False):
=======
    def _compute_beam_search_log_likelihood(jet, beam_size = 5, N_best=1, FullTree=False):
>>>>>>> Stashed changes

        startTime = time.time()
        # n = len(jet["leaves"])
        bs_jets = beam_search.recluster(
            jet,
            # beamSize=min(beam_size, n * (n - 1) // 2),
            beamSize= beam_size,
            delta_min=jet["pt_cut"],
            lam=float(jet["Lambda"]),
            N_best=N_best,
            visualize=False,
        )

        Time = time.time() - startTime

        bs_jet =bs_jets[0]

        if FullTree:
            return bs_jets
<<<<<<< Updated upstream

        elif MLEtree:
            return bs_jet
=======
>>>>>>> Stashed changes

        else:
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

        return [map_energy, Time,Z]


    @staticmethod
    def _compute_a_star_log_likelihood(tree, max_nodes = None, max_leaves=11):

        if len(tree["leaves"]) > max_leaves:
            return np.nan

        gt_jet = tree
        startTime = time.time()

        trellis = IterJetTrellis(leaves=gt_jet['leaves'],
                                 propagate_values_up=FLAGS.propagate_values_up,
                                 max_nodes=max_nodes,
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

        return [ - MAP_value, Time, step, trellis.nodes_explored, trellis.Ntrees]


    # @staticmethod
    def _compute_approx_a_star_log_likelihood(self, tree, beam_size, exact_heuristic_proof = False, approx_heuristic=False, max_nodes = None,max_leaves=11):

        # if len(tree["leaves"]) > max_leaves:
        #     return np.nan

        gt_jet = tree
        logging.info("Truth Tree = %s", gt_jet)
        startTime = time.time()

        trellis = ApproxIterJetTrellis(exact_heuristic_proof = exact_heuristic_proof,
                                        approx_heuristic = approx_heuristic,
                                       leaves=gt_jet['leaves'],
                                       propagate_values_up=FLAGS.propagate_values_up,
                                       max_nodes=max_nodes,
                                       min_invM=gt_jet['pt_cut'],
                                       Lambda=gt_jet['Lambda'],
                                       LambdaRoot=gt_jet['LambdaRoot'])

        BeamSearchTrees = self._compute_beam_search_log_likelihood(gt_jet, beam_size=beam_size, N_best=beam_size,  FullTree=True)
        logging.info("Number of Beam Search Trees= %s", len(BeamSearchTrees))

        for BeamSearchTree in BeamSearchTrees[0: FLAGS.maxBeamJets]:
            trellis.Add_BeamSearchHC( BeamSearchTree,
                         BeamSearchTree["root_id"],
                         trellis.clusters[trellis.root])


        hc, MAP_value, step = trellis.execute_search(num_matches=FLAGS.num_repeated_map_values,
                                                     max_steps=int(FLAGS.max_steps),
                                                     all_pairs_max_size = int(FLAGS.all_pairs_max_size),
                                                     num_tries = list(self.num_tries))



        Time = time.time() - startTime
        logging.info(f'total time = {Time}')
        logging.info(f'FINAL f ={MAP_value}')
        logging.info("-------------------------------------------")

        return [ - MAP_value, Time, step, trellis.nodes_explored, trellis.Ntrees]


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
    os.system("mkdir -p "+FLAGS.output_dir)
    max_nodes = 2 ** FLAGS.NleavesMin + 10
    # beam_size = 3 * FLAGS.NleavesMin
    beam_size = int(np.minimum(np.maximum(3 * FLAGS.NleavesMin, FLAGS.NleavesMin * (FLAGS.NleavesMin - 1) / 2),1000))
    # beam_size = 1000

    num_tries = np.asarray([FLAGS.num_tries_m, FLAGS.num_tries_k]).flatten()
    logging.info("Num tries =%s", num_tries)

    Evaluator = GinkgoEvaluator(num_tries=num_tries)

    Evaluator.eval_true("truth")


    if FLAGS.algorithm =="BeamSearchTrees":
        """Save file with all MLE beam search trees"""
        logging.info("Beam Size =%s", beam_size)
        Evaluator.eval_beam_search("BS", beam_size = beam_size, MLEtree=True)

        # logging.info("First tree =%s", Evaluator.BStrees[0])
        out_dir = os.path.join(FLAGS.output_dir, "BeamSearchTrees")
        os.system("mkdir -p "+out_dir)

        filename = FLAGS.results_filename
        out_file = os.path.join(out_dir,filename)
        if os.path.exists(out_dir):
            with open(out_file, "wb") as f:
                pickle.dump(Evaluator.BStrees, f, protocol=2)


    if FLAGS.algorithm =="BeamSearchGreedy":
        logging.info("Beam Size =%s", beam_size)
        Evaluator.eval_greedy("Greedy")
        Evaluator.eval_beam_search("BS", beam_size = beam_size)

        out_dir = os.path.join(FLAGS.output_dir, "BeamSearchGreedy")
        os.system("mkdir -p "+out_dir)
        Evaluator.save(out_dir)


    if FLAGS.algorithm =="BeamSearchGreedy1000":
        beam_size = 1000
        logging.info("Beam Size =%s", beam_size)
        Evaluator.eval_greedy("Greedy")
        Evaluator.eval_beam_search("BS", beam_size = beam_size)

        out_dir = os.path.join(FLAGS.output_dir, "BeamSearchGreedy1000")
        os.system("mkdir -p "+out_dir)
        Evaluator.save(out_dir)

    if FLAGS.algorithm == "ExactTrellis":
        Evaluator.eval_exact_trellis("exact_trellis", max_leaves=FLAGS.max_leaves)

        out_dir = os.path.join(FLAGS.output_dir, "ExactTrellis")
        os.system("mkdir -p "+out_dir)
        Evaluator.save(out_dir)

    if FLAGS.algorithm == "ExactAstar":
        Evaluator.eval_exact_a_star("exact_a_star", max_leaves=FLAGS.max_leaves,  max_nodes=max_nodes)

        out_dir = os.path.join(FLAGS.output_dir, "ExactAstar")
        os.system("mkdir -p "+out_dir)
        Evaluator.save(out_dir)


    if FLAGS.algorithm == "ApproxAstar":

        if FLAGS.NleavesMin >22:
            max_nodes = FLAGS.max_nodes
        logging.info("Beam Size =%s", beam_size)
        logging.info(" max_nodes =%s", max_nodes)
        details={"NumTries":Evaluator.num_tries,"all_pairs_max_size":FLAGS.all_pairs_max_size, "max_steps":FLAGS.max_steps}
        Evaluator.eval_approx_a_star("approx_a_star",exact_heuristic_proof = FLAGS.exact_heuristic_proof, approx_heuristic=FLAGS.approx_heuristic, beam_size = beam_size, max_nodes =max_nodes, info=details)

        if FLAGS.exact_heuristic_proof:
            out_dir = os.path.join(FLAGS.output_dir, "ApproxAstarExactHeuristicProof")

        elif FLAGS.approx_heuristic==False:

            if FLAGS.maxBeamJets>1:
                out_dir = os.path.join(FLAGS.output_dir, "ApproxAstarFullBeam")
            else:
                out_dir = os.path.join(FLAGS.output_dir, "ApproxAstar")

        else:
            if FLAGS.exact_astar_approx_heuristic:
                out_dir = os.path.join(FLAGS.output_dir, "ExactAstarApproxHeuristic")

            else:
                out_dir = os.path.join(FLAGS.output_dir, "ApproxAstarApproxHeuristicFullBeam")

        os.system("mkdir -p "+out_dir)
        Evaluator.save(out_dir, ApproxAstar=True)



    logging.info("Tree size =%s",Evaluator.tree_size)
    logging.info("log_likelihoods= %s", Evaluator.log_likelihoods)
    logging.info("Illegal actions =  %s" , Evaluator.illegal_actions)
    logging.info("Times = %s ",   Evaluator.times)
    logging.info("Number of evaluations =  %s", Evaluator.likelihood_evaluations)
    logging.info("saving output in %s", str(out_dir))

if __name__ == '__main__':
    app.run(main)

# sys.exit()