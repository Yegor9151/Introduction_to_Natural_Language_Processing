import tensorflow as tf

def create_padding_mask(seq):
    """
    create_padding_mask(seq=tf.constant([[7, 6, 0, 0, 1], 
                                         [1, 2, 3, 0, 0], 
                                         [0, 0, 0, 4, 5]]))
                                         
    <tf.Tensor: shape=(3, 1, 1, 5), dtype=float32, numpy=
    array([[[[0., 0., 1., 1., 0.]]],
           [[[0., 0., 0., 1., 1.]]],
           [[[1., 1., 1., 0., 0.]]]], dtype=float32)>
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = seq[:, tf.newaxis, tf.newaxis, :]
    return seq

def create_look_ahead_mask(size):
    """
    create_look_ahead_mask(size=tf.random.uniform((1, 3)).shape[1])
    
    out:
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[0., 1., 1.],
           [0., 0., 1.],
           [0., 0., 0.]], dtype=float32)>
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = dec_padding_mask = create_padding_mask(inp)
    
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_target_padding_mask = create_padding_mask(tar)
    
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask