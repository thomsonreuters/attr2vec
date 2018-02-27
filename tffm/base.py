import tensorflow as tf
from .core import TFFMCore
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
import six
from tqdm import tqdm
import numpy as np
import os
from tensorflow.contrib.tensorboard.plugins import projector
import sys


def batcher(input_X, input_X_ids, input_X_weights, input_y, perm,  batch_size=-1):
    """Split data to mini-batches.

    Parameters
    ----------
    X_ : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y_ : np.array or None, shape (n_samples,)
        Target vector relative to X.

    batch_size : int
        Size of batches.
        Use -1 for full-size batches

    Yields
    -------
    ret_x : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Same type as input
    ret_y : np.array or None, shape (batch_size,)
    """
    n_samples = input_y.shape[0]
    n_fields = len(input_X_ids)

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))


    # shuffle the data
    X = input_X[perm]
    X_ids = []
    X_weights = []
    y = input_y[perm]
    for m in range(n_fields):
        X_ids.append(input_X_ids[m][perm])
        X_weights.append(input_X_weights[m][perm])


    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)

        ret_x = X[i:upper_bound]
        ret_y = y[i:upper_bound]
        ret_x_ids = []
        ret_x_weights = []

        for m in range(n_fields):
            ret_x_ids.append(X_ids[m][i:upper_bound])
            ret_x_weights.append(X_weights[m][i:upper_bound])
        
        yield (ret_x, ret_x_ids, ret_x_weights, ret_y)


def batch_to_feeddict(X, x_ids, x_weights, y, core):
    """Prepare feed dict for session.run() from mini-batch.
    Convert sparse format into tuple (indices, values, shape) for tf.SparseTensor
    Parameters
    ----------
    X : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Training vector, where batch_size in the number of samples and
        n_features is the number of features.
    y : np.array, shape (batch_size,)
        Target vector relative to X.
    core : TFFMCore
        Core used for extract appropriate placeholders
    Returns
    -------
    fd : dict
        Dict with formatted placeholders
    """
    fd = {}
    if core.input_type == 'dense':
        fd[core.train_x] = X.astype(np.float32)

    else:

        # sparse case
        X_sparse = X.tocoo()
        fd[core.raw_indices] = np.hstack(
            (X_sparse.row[:, np.newaxis], X_sparse.col[:, np.newaxis])
        ).astype(np.int64)
        fd[core.raw_values] = X_sparse.data.astype(np.float32)
        fd[core.raw_shape] = np.array(X_sparse.shape).astype(np.int64)

        for m in range(len(x_ids)):

            local_x_ids = x_ids[m].tocoo()
            local_x_weights = x_weights[m].tocoo()

            fd[core.raw_indices_array[m]] = np.hstack((local_x_ids.row[:, np.newaxis], local_x_ids.col[:, np.newaxis])).astype(np.int64)
            fd[core.raw_ids_array[m]] = local_x_ids.data.astype(np.float32)
            fd[core.raw_weights_array[m]] = local_x_weights.data.astype(np.float32)
            fd[core.raw_shape_array[m]] = np.array(local_x_ids.shape).astype(np.int64)


    fd[core.train_y] = y.astype(np.float32)
    return fd


class TFFMBaseModel(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for FM.
    This class implements L2-regularized arbitrary order FM model.

    It supports arbitrary order of interactions and has linear complexity in the
    number of features (a generalization of the approach described in Lemma 3.1
    in the referenced paper, details will be added soon).

    It can handle both dense and sparse input. Only numpy.array and CSR matrix are
    allowed as inputs; any other input format should be explicitly converted.

    Support logging/visualization with TensorBoard.


    Parameters (for initialization)
    ----------
    batch_size : int, default: -1
        Number of samples in mini-batches. Shuffled every epoch.
        Use -1 for full gradient (whole training set in each batch).

    n_epoch : int, default: 100
        Default number of epoches.
        It can be overrived by explicitly provided value in fit() method.

    log_dir : str or None, default: None
        Path for storing model stats during training. Used only if is not None.
        WARNING: If such directory already exists, it will be removed!
        You can use TensorBoard to visualize the stats:
        `tensorboard --logdir={log_dir}`

    session_config : tf.ConfigProto or None, default: None
        Additional setting passed to tf.Session object.
        Useful for CPU/GPU switching, setting number of threads and so on,
        `tf.ConfigProto(device_count = {'GPU': 0})` will disable GPU (if enabled)

    verbose : int, default: 0
        Level of verbosity.
        Set 1 for tensorboard info only and 2 for additional stats every epoch.

    kwargs : dict, default: {}
        Arguments for TFFMCore constructor.
        See TFFMCore's doc for details.

    Attributes
    ----------
    core : TFFMCore or None
        Computational graph with internal utils.
        Will be initialized during first call .fit()

    session : tf.Session or None
        Current execution session or None.
        Should be explicitly terminated via calling destroy() method.

    steps : int
        Counter of passed lerning epochs, used as step number for writing stats

    n_features : int
        Number of features used in this dataset.
        Inferred during the first call of fit() method.

    intercept : float, shape: [1]
        Intercept (bias) term.

    weights : array of np.array, shape: [order]
        Array of underlying representations.
        First element will have shape [n_features, 1],
        all the others -- [n_features, rank].

    Notes
    -----
    You should explicitly call destroy() method to release resources.
    See TFFMCore's doc for details.
    """


    def init_basemodel(self, num_features, n_epochs=100, batch_size=-1, log_dir=None, session_config=None, verbose=0, seed=None, words=[], write_embedding_every = 1, **core_arguments):
        core_arguments['seed'] = seed
        self.core = TFFMCore(**core_arguments)
        self.core.set_num_features(num_features)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        self.session_config = session_config
        self.verbose = verbose
        self.steps = 0
        self.seed = seed
        self.words = words
        self.write_embedding_every = write_embedding_every
        
    def initialize_session(self):
        """Start computational session on builded graph.
        Initialize summary logger (if needed).
        """
        if self.core.graph is None:
            raise 'Graph not found. Try call .core.build_graph() before .initialize_session()'
        if self.need_logs:
            self.summary_writer = tf.summary.FileWriter(self.log_dir, self.core.graph)
            if self.verbose > 0:
                full_log_path = os.path.abspath(self.log_dir)
                print('Initialize logs, use: \ntensorboard --logdir={}'.format(full_log_path))
        self.session = tf.Session(config=self.session_config, graph=self.core.graph)
        self.session.run(self.core.init_all_vars)

    @abstractmethod
    def preprocess_target(self, target):
        """Prepare target values to use."""

    def fit(self, X, X_ids, X_weights, y_, n_epochs=None, show_progress=False):

        assert(len(X_ids) == len(X_weights))
        for m in range(len(X_ids)):
            assert(X_ids[m].shape[0] == X_weights[m].shape[0] == y_.shape[0] == X.shape[0])

        if self.core.n_features is None:
            self.core.set_num_features(X.shape[1])

        assert self.core.n_features==X.shape[1], 'Different num of features in initialized graph and input'

        if self.core.graph is None:
            self.core.build_graph()
            self.initialize_session()

        used_y = self.preprocess_target(y_)

        if n_epochs is None:
            n_epochs = self.n_epochs

        # For reproducible results
        if self.seed:
            np.random.seed(self.seed)

        # Training cycle
        for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not show_progress)):
            # generate permutation
            perm = np.random.permutation(X.shape[0])
            epoch_loss = []
            # iterate over batches

            for ret_x, ret_x_ids, ret_x_weights, ret_y in batcher(X, X_ids, X_weights, y_, perm, batch_size=self.batch_size):




                fd = batch_to_feeddict(ret_x, ret_x_ids, ret_x_weights, ret_y, core=self.core)
                ops_to_run = [self.core.trainer, self.core.target, self.core.summary_op]
                result = self.session.run(ops_to_run, feed_dict=fd)
                _, batch_target_value, summary_str = result
                epoch_loss.append(batch_target_value)

                # write percentage
                num_steps = int((np.modf(X.shape[0]/self.batch_size)[1]+1)*(epoch+1))                
                sys.stdout.write("\r\tstep {} / {}, loss {:g}"
                                         .format(self.steps, num_steps, batch_target_value))
                sys.stdout.flush()


                '''A,B,C,all_AB = self.session.run([self.core.A,self.core.part_1,self.core.part_2,self.core.all_AB], feed_dict=fd)
                print(A.shape)
                print(B.shape)
                print(C.shape)
                print(all_AB.shape)'''


                #print(" ")
                #print("REGLARIZATION {}".format(regularization))
                #print(" ")

                # write stats 
                if self.need_logs:
                    self.summary_writer.add_summary(summary_str, self.steps)
                    self.summary_writer.flush()
                    
                self.steps += 1

            if (epoch != 0) and ((epoch % self.write_embedding_every) == 0):
                print('writing embeddings on file')
                with self.session.as_default():
                    embedding_path = self.log_dir+"/vectors_e"+str(epoch)+".txt"
                    embedding_file = open(embedding_path, 'w')
                    vectors = self.core.embeddings_2.eval()
                    i = 0
                    for vector in vectors:  
                        embedding_file.write(str(self.words[i])+" ")
                        for value in vector:
                            embedding_file.write(str(value)+" ")
                        embedding_file.write('\n')
                        i+=1
                    embedding_file.close()


            if self.verbose > 1:

                    print('[epoch {}]: mean target value: {}'.format(epoch, np.mean(epoch_loss)))

                    with self.session.as_default():

                        # print norms
                        try:
                            meta_vector = self.core.meta.eval()
                            '''vectors_norm = self.core.vectors_norm_sqrt.eval()
                            for i in range(0,len(self.core.valid_examples)):
                                idx = self.core.valid_examples[i]
                                valid_word = self.words[idx]
                                valid_norm = vectors_norm[idx]
                                reweight = reweights[idx]
                                meta = meta_vector[idx]
                                print("{} - {} - {} - {} ".format(valid_word,valid_norm, reweight, meta))
                            print(" ")'''

                            # print similarity
                            print(self.core.n_features)
                            print(' ')  
                            sim = self.core.similarity.eval()
                            for i in range(0,len(self.core.valid_examples)):
                                valid_word = self.words[self.core.valid_examples[i]]
                                top_k = 10  # number of nearest neighbors
                                upper_bound_top_k = 1000
                                nearest = (-sim[i, :]).argsort()[1:upper_bound_top_k + 1]
                                log_str = "Nearest to %s:" % valid_word

                                k_words = 0
                                k = 0
                                while (k_words<top_k):
                                    if (meta_vector[nearest[k]] == 0): # consider only words
                                        close_word = self.words[nearest[k]]
                                        log_str = "%s %s," % (log_str, close_word)
                                        k_words += 1
                                    k += 1
                                print(log_str)
                            print()
                        except:
                            pass

                        '''tot_sim = 0.0     
                        valid_words = [ self.words[index] for index in self.core.valid_examples ]

                        print("  -  \t"+"\t".join(valid_words))
                        for i in range(0,len(self.core.valid_examples)):
                            word_i = self.words[self.core.valid_examples[i]]
                            log_str = str(word_i)+"\t"
                            for j in range(0,len(self.core.valid_examples)):
                                index_word_j = self.core.valid_examples[j]
                                word_j = self.words[index_word_j]                            
                                sim_i_j = sim[i, index_word_j]
                                log_str = "%s  %g" % (log_str, sim_i_j)
                                tot_sim += sim_i_j
                            print(log_str)

                        print()
                        print("tot_sim: "+str(tot_sim))
                        print()'''

            if self.need_logs:
                with self.session.as_default():
                    self.core.saver.save(self.session, self.log_dir+"/model.ckpt")
                    #TensorBoard: Embedding Visualization
                    # Use the same LOG_DIR where you stored your checkpoint.
                    #summary_writer = tf.train.SummaryWriter(LOG_DIR)
                    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
                    config = projector.ProjectorConfig()
                    # You can add multiple embeddings. Here we add only one.

                    embedding_2 = config.embeddings.add()
                    embedding_2.tensor_name = self.core.embeddings_2.name
                    embedding_2.metadata_path = "metadata.tsv"

                    # embedding_3 = config.embeddings.add()
                    # embedding_3.tensor_name = self.core.embeddings_3.name
                    # embedding_3.metadata_path = "log/metadata.tsv"

                    # Link this tensor to its metadata file (e.g. labels).
                    
                    # Saves a configuration file that TensorBoard will read during startup.
                    projector.visualize_embeddings(self.summary_writer, config)

        # write embeddings on file
        print('writing embeddings on file')
        with self.session.as_default():
            embedding_path = self.log_dir+"/vectors.txt"
            embedding_file = open(embedding_path, 'w')
            vectors = self.core.embeddings_2.eval()
            i = 0
            for vector in vectors:  
                embedding_file.write(str(self.words[i])+" ")
                for value in vector:
                    embedding_file.write(str(value)+" ")
                embedding_file.write('\n')
                i+=1
            embedding_file.close()


    def decision_function(self, X):
        if self.core.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for bX, bY in batcher(X, y_=None, batch_size=self.batch_size):
            fd = batch_to_feeddict(bX, bY, core=self.core)
            output.append(self.session.run(self.core.outputs, feed_dict=fd))
        distances = np.concatenate(output).reshape(-1)
        # WARN: be carefull with this reshape in case of multiclass
        return distances

    @abstractmethod
    def predict(self, X):
        """Predict target values for X."""

    @property
    def intercept(self):
        """Export bias term from tf.Variable to float."""
        return self.core.b.eval(session=self.session)

    @property
    def weights(self):
        """Export underlying weights from tf.Variables to np.arrays."""
        return [x.eval(session=self.session) for x in self.core.w]

    def save_state(self, path):
        self.core.saver.save(self.session, path)

    def load_state(self, path):
        if self.core.graph is None:
            self.core.build_graph()
            self.initialize_session()
        self.core.saver.restore(self.session, path)

    def destroy(self):
        """Terminates session and destroyes graph."""
        self.session.close()
        self.core.graph = None
