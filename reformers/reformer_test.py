from reformers import TFReformerLM
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0],True)
model_tf = TFReformerLM(
    num_tokens= 1500,
    emb = 512,
    depth = 1,
    max_seq_len = 320000,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 512,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    use_full_attn = False   # use full self attention, for comparison
)

x = tf.random.uniform((1, 320000))
model_tf.build(input_shape=(1,320000))
model_tf.summary()
y = model_tf(x)

