import tensorflow as tf
from tqdm import tqdm
from tffm import TFFMRegressor
from reader import Reader

log_path = "log/"
metadata_path = log_path+"metadata.tsv"
word_meta2_id_file = 'data_pos/Word2Id.csv' #'data_dependency/Word2Id.csv'
cooccur_file = 'data_pos/Cooccur.csv' #'data_dependency/Cooccur.csv'
iterations = 500
dimensions = 200
lr = 0.05 # 0.001
batch_size = 500000 #131072 #32768
count_max = 100

# READ INPUT
reader = Reader(word_meta2_id_file,cooccur_file,metadata_path)
vocab_size = reader.vocab_size
len_unique_meta = reader.len_unique_meta
valid_examples_words = reader.valid_examples_words
#valid_examples_pos = reader.valid_examples_pos
words = reader.words
meta_vector = reader.meta_vector
X = reader.X
X_ids = reader.X_ids
X_weights = reader.X_weights
Y = reader.Y

model = TFFMRegressor(
    num_unique_meta=len_unique_meta,
    meta_vector=meta_vector,
    num_features=vocab_size,
    order=2, 
    rank=dimensions, 
    # optimizer=tf.train.AdamOptimizer(learning_rate=lr),   # lr = 0.001
    optimizer=tf.train.AdagradOptimizer(learning_rate=lr),  # lr = 0.05
    n_epochs=iterations, 
    batch_size=batch_size,
    init_std=0.01,
    reg=0.02,
    reweight_reg=False,
    count_max=count_max,
    input_type='sparse',
    log_dir=log_path,
    valid_examples=valid_examples_words,
    words=words,
    write_embedding_every=10,
    session_config=tf.ConfigProto(log_device_placement=False), 
    verbose=2
)
model.fit(X, X_ids, X_weights, Y, show_progress=True)