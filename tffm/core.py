import tensorflow as tf
from . import utils
import math


class TFFMCore():
    """
    This class implements underlying routines about creating computational graph.

    Its required `n_features` to be set at graph building time.


    Parameters
    ----------
    order : int, default: 2
        Order of corresponding polynomial model.
        All interaction from bias and linear to order will be included.

    rank : int, default: 5
        Number of factors in low-rank appoximation.
        This value is shared across different orders of interaction.

    input_type : str, 'dense' or 'sparse', default: 'dense'
        Type of input data. Only numpy.array allowed for 'dense' and
        scipy.sparse.csr_matrix for 'sparse'. This affects construction of
        computational graph and cannot be changed during training/testing.

    loss_function : function: (tf.Op, tf.Op) -> tf.Op, default: None
        Loss function.
        Take 2 tf.Ops: outputs and targets and should return tf.Op of loss
        See examples: .utils.loss_mse, .utils.loss_logistic

    optimizer : tf.train.Optimizer, default: AdamOptimizer(learning_rate=0.01)
        Optimization method used for training

    reg : float, default: 0
        Strength of L2 regularization

    use_diag : bool, default: False
        Use diagonal elements of weights matrix or not.
        In the other words, should terms like x^2 be included.
        Ofter reffered as a "Polynomial Network".
        Default value (False) corresponds to FM.

    reweight_reg : bool, default: False
        Use frequency of features as weights for regularization or not.
        Should be usefull for very sparse data and/or small batches

    init_std : float, default: 0.01
        Amplitude of random initialization

    seed : int or None, default: None
        Random seed used at graph creating time


    Attributes
    ----------
    graph : tf.Graph or None
        Initialized computational graph or None

    trainer : tf.Op
        TensorFlow operation node to perform learning on single batch

    n_features : int
        Number of features used in this dataset.
        Inferred during the first call of fit() method.

    saver : tf.Op
        tf.train.Saver instance, connected to graph

    summary_op : tf.Op
        tf.merge_all_summaries instance for export logging

    b : tf.Variable, shape: [1]
        Bias term.

    w : array of tf.Variable, shape: [order]
        Array of underlying representations.
        First element will have shape [n_features, 1],
        all the others -- [n_features, rank].

    Notes
    -----
    Parameter `rank` is shared across all orders of interactions (except bias and
    linear parts).
    tf.sparse_reorder doesn't requied since COO format is lexigraphical ordered.
    This implementation uses a generalized approach from referenced paper along
    with caching.

    References
    ----------
    Steffen Rendle, Factorization Machines
        http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """
    def __init__(self, num_unique_meta, meta_vector, order=2, rank=2, input_type='dense', loss_function=utils.loss_logistic, 
                optimizer=tf.train.AdamOptimizer(learning_rate=0.01), reg=0, init_std=0.1, 
                use_diag=False, reweight_reg=False, seed=None, count_max = 100, scaling_factor = 0.75, valid_examples = [] ):
        self.meta_vector = meta_vector
        self.num_unique_meta = num_unique_meta
        print("self.num_unique_meta: {}".format(self.num_unique_meta))
        self.order = order
        self.rank = rank
        self.use_diag = use_diag
        self.input_type = input_type
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.reg = reg
        self.reweight_reg = reweight_reg
        self.init_std = init_std
        self.seed = seed
        self.n_features = None
        self.graph = None
        self.count_max = count_max
        self.scaling_factor = scaling_factor
        self.valid_examples = valid_examples

    def set_num_features(self, n_features):
        self.n_features = n_features

    def init_learnable_params(self):
        r = self.rank

        self.meta = tf.constant(self.meta_vector, name="meta_vector")

        if isinstance(self.reg, list):
            self.meta_indices = [None] * len(self.reg)
            # regularization meta specific
            for m in range(0, len(self.reg)):
                node_name = 'meta_indices_' + str(m)
                meta_value = tf.constant(m, dtype=tf.int32)
                self.meta_indices[m] = tf.where(tf.equal(self.meta, meta_value), name=node_name)

        # to visualize embeddings of order 1 = biases
        # self.biases = tf.Variable(tf.zeros([self.n_features, 1], tf.float32), trainable=False, name="biases")
        self.biases = tf.Variable(tf.random_uniform([self.n_features, 1], -self.init_std, self.init_std), name="biases")

        # to visualize embeddings of order 2 = word_embeddings
        #self.embeddings_2 = tf.Variable(tf.random_uniform([self.n_features, self.rank], -self.init_std, self.init_std), name="word_embedding")
        self.embeddings_2 = tf.Variable(tf.truncated_normal([self.n_features, self.rank], stddev=self.init_std), name="word_embedding")
        

        self.b = tf.Variable(self.init_std, trainable=True, name='bias')
        #self.b = tf.Variable(0.0, trainable=False, name='bias')

        #collect some summaries
        tf.summary.histogram("biases", self.biases)
        tf.summary.histogram("embeddings_2", self.embeddings_2)
        tf.summary.scalar('bias', self.b)
        tf.summary.scalar('self.n_features', self.n_features)


    def init_similarity_computation(self):        
        self.vectors_norm_sqrt = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings_2), 1, keep_dims=True))
        normalized_embeddings = self.embeddings_2 / self.vectors_norm_sqrt
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True, name="cosine_similarity")


    def init_placeholders(self):

        self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32, name="similarity_examples") 

        self.train_ids = []
        self.train_weights = []

        if self.input_type == 'dense':
            self.train_x = tf.placeholder(tf.float32, shape=[None, self.n_features], name='x')
        else:
            with tf.name_scope('sparse_placeholders') as scope:
                self.raw_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
                self.raw_values = tf.placeholder(tf.float32, shape=[None], name='raw_data')
                self.raw_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')     

                self.raw_indices_array = []
                self.raw_ids_array = []
                self.raw_weights_array = []
                self.raw_shape_array = []

                for meta_index in range(self.num_unique_meta):

                    local_raw_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices_'+str(meta_index))
                    local_raw_ids = tf.placeholder(tf.int32, shape=[None], name='raw_ids_'+str(meta_index))
                    local_raw_weights = tf.placeholder(tf.float32, shape=[None], name='raw_weights_'+str(meta_index))
                    local_raw_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape_'+str(meta_index))

                    self.raw_indices_array.append(local_raw_indices)
                    self.raw_ids_array.append(local_raw_ids)
                    self.raw_weights_array.append(local_raw_weights)
                    self.raw_shape_array.append(local_raw_shape)

                    local_train_ids = tf.SparseTensor(local_raw_indices, local_raw_ids, local_raw_shape)
                    local_train_weights = tf.SparseTensor(local_raw_indices, local_raw_weights, local_raw_shape)

                    self.train_ids.append(tf.sparse_tensor_to_dense(local_train_ids,default_value=-1,validate_indices=False))
                    self.train_weights.append(tf.sparse_tensor_to_dense(local_train_weights,default_value=-1,validate_indices=False))

            # tf.sparse_reorder is not needed since scipy return COO in canonical order
            self.train_x = tf.SparseTensor(self.raw_indices, self.raw_values, self.raw_shape)

        self.train_y = tf.placeholder(tf.float32, shape=[None], name='Y')

    def pow_matmul(self, order, pow):
        if pow not in self.x_pow_cache:
            x_pow = utils.pow_wrapper(self.train_x, pow, self.input_type)
            self.x_pow_cache[pow] = x_pow
        if order not in self.matmul_cache:
            self.matmul_cache[order] = {}
        if pow not in self.matmul_cache[order]:

            w_pow = tf.pow(self.embeddings_2, pow)
            dot = utils.matmul_wrapper(self.x_pow_cache[pow], w_pow, self.input_type)
            self.matmul_cache[order][pow] = dot
        return self.matmul_cache[order][pow]

    def init_main_block(self):
        self.x_pow_cache = {}
        self.matmul_cache = {}
        self.outputs = self.b

        #LINEAR PART
        with tf.name_scope('linear_part') as scope:
            contribution = utils.matmul_wrapper(self.train_x, self.biases, self.input_type)
        self.outputs += contribution

        #ORIGINAL CODE
        i = 2
        with tf.name_scope('order_{}'.format(i)) as scope:

             # word embedding
            raw_dot = utils.matmul_wrapper(self.train_x, self.embeddings_2, self.input_type)

            dot = tf.pow(raw_dot, i)
            if self.use_diag:
                contribution = tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
                contribution /= 2.0**(i-1)
            else:
                initialization_shape = tf.shape(dot)
                for in_pows, out_pows, coef in utils.powers_and_coefs(i):
                    product_of_pows = tf.ones(initialization_shape)
                    for pow_idx in range(len(in_pows)):
                        pmm = self.pow_matmul(i, in_pows[pow_idx])
                        product_of_pows *= tf.pow(pmm, out_pows[pow_idx])
                    dot -= coef * product_of_pows
                contribution = tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
                contribution /= float(math.factorial(i))
            self.outputs += contribution

        # FACTORIZATION MACHINES ORDER 2
        '''self.A = tf.nn.embedding_lookup(self.embeddings_2, self.train_ids, name="A")
        self.part_1 = tf.pow(tf.reduce_sum(self.A, 1, name="B"),2)
        self.part_2 = tf.reduce_sum(tf.pow(self.A,2), 1, name="C")
        self.all_AB = tf.expand_dims(tf.divide(tf.reduce_sum(tf.subtract(self.part_1,self.part_2), 1, name="D"),2), 1)

        #product = tf.expand_dims(tf.reduce_sum(tf.squeeze(tf.multiply(A, B, name="mult"), 1), 1, name="embeddings_product"), 1)
        self.outputs += self.all_AB'''

        # CUSTOM OPTIMIZED CODE
        '''field1_embeddings = tf.nn.embedding_lookup(self.embeddings_2, self.train_ids[0], name="field1_embeddings")
        field2_embeddings = tf.nn.embedding_lookup(self.embeddings_2, self.train_ids[1], name="field2_embeddings")
        mul_w2_p2 = tf.expand_dims(tf.reduce_sum(tf.squeeze(tf.multiply(field1_embeddings, field2_embeddings, name="mult"), 1), 1, name="embeddings_product"), 1)
        self.outputs += mul_w2_p2'''



    def init_regularization(self):
        with tf.name_scope('regularization') as scope:
            self.regularization = 0
            
            with tf.name_scope('reweights') as scope:

                if self.reweight_reg:
                    counts = utils.count_nonzero_wrapper(self.train_x, self.input_type)
                    sqrt_counts = tf.transpose(tf.sqrt(tf.to_float(counts)))
                else:
                    sqrt_counts = tf.ones_like(self.biases)                    
                self.reweights = sqrt_counts

            
            order = 2

            if isinstance(self.reg, list):
                # regularization meta specific

                vectors_norm = tf.reduce_sum(tf.square(self.embeddings_2*self.reweights), 1, keep_dims=True)      

                for m in range(0, len(self.reg)):
                    node_name = 'regularization_penalty_' + str(order) + "_m_" + str(m)
                    meta_norm = tf.sqrt(tf.reduce_sum(
                        tf.nn.embedding_lookup(vectors_norm, self.meta_indices[m])
                        ), name=node_name)
                    increment = self.reg[m] * meta_norm
                    tf.summary.scalar('penalty_W_{}_meta_{}'.format(order,m), increment)
                    self.regularization += increment
            else:
                self.reweights = sqrt_counts / tf.reduce_sum(sqrt_counts)
                node_name = 'regularization_penalty_' + str(order)
                # word embedding
                norm = tf.reduce_mean(tf.pow(self.embeddings_2*self.reweights, 2), name=node_name)
                tf.summary.scalar('penalty_W_{}'.format(order), norm)
                self.regularization += self.reg * norm

            tf.summary.histogram("regularization_penalty", self.regularization)            
            tf.summary.scalar('regularization_penalty', self.regularization)

    def init_loss(self):
        with tf.name_scope('loss') as scope:

            # old
            # self.loss = self.loss_function(self.outputs, self.train_y)
            # self.reduced_loss = tf.reduce_mean(self.loss)

            # Glove style    

            log_cooccurrences = tf.log(tf.to_float(self.train_y), name="log_cooccurrences")
            negative_log_cooccurrences = tf.expand_dims(tf.negative(log_cooccurrences), -1)

            # weighting_factor
            div_count_max = tf.div(self.train_y, self.count_max)
            pow_count_max = tf.pow(div_count_max,self.scaling_factor)
            self.weighting_factor = tf.expand_dims(tf.minimum(1.0 , pow_count_max, name="weighting_factor"), -1)

            self.distance_expr = tf.square( 
                tf.add_n([self.outputs,
                          negative_log_cooccurrences
                          ]) , name="distance_expr")

            single_losses = tf.multiply(self.weighting_factor, self.distance_expr, name="single_losses")
            self.reduced_loss = tf.reduce_sum(single_losses, name="total_loss")

            '''tf.summary.histogram("log_cooccurrences", log_cooccurrences)
            tf.summary.histogram("div_count_max", div_count_max)
            tf.summary.histogram("pow_count_max", pow_count_max)
            tf.summary.histogram("weighting_factor", self.weighting_factor)
            tf.summary.histogram("self.train_y", self.train_y)
            tf.summary.histogram("negative_log_cooccurrences", negative_log_cooccurrences)
            tf.summary.histogram("self.outputs", self.outputs)
            tf.summary.histogram("distance_expr", self.distance_expr)
            tf.summary.histogram("single_losses", single_losses)
            tf.summary.histogram("self.reduced_loss", self.reduced_loss)
            tf.summary.histogram("weighting_factor", self.weighting_factor)
            tf.summary.scalar('loss', self.reduced_loss)'''

    def init_target(self):
        with tf.name_scope('target') as scope:
            self.target = self.reduced_loss + self.regularization
            '''self.checked_target = tf.verify_tensor_all_finite(
                self.target,
                msg='NaN or Inf in target value', 
                name='target')
            tf.summary.scalar('target', self.checked_target)'''

    def build_graph(self):
        """Build computational graph according to params."""
        assert self.n_features is not None, 'Number of features is unknown. It can be set explicitly by .core.set_num_features'
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            with tf.name_scope('learnable_params') as scope:
                self.init_learnable_params()
            with tf.name_scope('input_block') as scope:
                self.init_placeholders()
            with tf.name_scope("cosine_similarity"):
                self.init_similarity_computation()
            with tf.name_scope('main_block') as scope:
                self.init_main_block()
            with tf.name_scope('optimization_criterion') as scope:
                self.init_regularization()
                self.init_loss()
                self.init_target()
            self.trainer = self.optimizer.minimize(self.target)
            self.init_all_vars = tf.global_variables_initializer()
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
