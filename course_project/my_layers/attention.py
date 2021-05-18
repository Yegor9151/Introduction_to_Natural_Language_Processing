import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


def scaled_dot_product_attention(q, k, v, mask):
    """
    k = tf.constant([[10,0,0],
                 [0,10,0],
                 [0,0,10],
                 [0,0,10]], dtype=tf.float32)  # (4, 3)

    v = tf.constant([[   1,0],
                     [  10,0],
                     [ 100,5],
                     [1000,6]], dtype=tf.float32)  # (4, 2)

    q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    
    scaled_dot_product_attention(q, k, v, None)
    
    out:
    (<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[1.000000e+01, 9.276601e-25]], dtype=float32)>,
     <tf.Tensor: shape=(1, 4), dtype=float32, numpy=
     array([[8.433274e-26, 1.000000e+00, 8.433274e-26, 8.433274e-26]],
           dtype=float32)>)
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    dk = tf.math.sqrt(dk)
    
    scaled_attention_logits = matmul_qk / dk

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(Layer):
    """
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    out.shape, attn.shape
    
    out: (TensorShape([1, 60, 512]), TensorShape([1, 8, 60, 60]))
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        self.dense = Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        output, weight = scaled_dot_product_attention(q, k, v, mask)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        return output, weight