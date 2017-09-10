import tensorflow as tf
from roadhouse.models.classifiers import zoo

def gram_matrix(layer, features_nr):
    flat = tf.reshape(layer, (-1, features_nr))

    gram = tf.matmul(flat, tf.transpose(flat))
    return gram


# indices - 0 is white noise image, 1 is content, 2 is style


def build_transferer(net_func, weights_name_file, sess, frames_nr, content_layers=None, style_layers=None):
    if content_layers is None:
        content_layers = ['conv_1', 'conv_2']

    if style_layers is None:
        style_layers = ['conv_3', 'pooled']

    init_song = tf.Variable(tf.truncated_normal([1, frames_nr, 513]))
    content_spectrogram = tf.placeholder('float', [1, frames_nr, 513])
    style_spectrogram = tf.placeholder('float', [1, frames_nr, 513])
    init_tensor = tf.concat([init_song, content_spectrogram, style_spectrogram], axis=0)
    model, layers = zoo[net_func](init_tensor, 10, True)

    recoverer = tf.train.Saver(var_list=[x for x in tf.trainable_variables()
                                         if not x.name.startswith('Variable')])
    recoverer.restore(sess=sess, save_path=weights_name_file)

    style_loss = 0.
    content_loss = 0.

    for layer_name in style_layers:
        try:
            _, length, features_nr = layers[layer_name].get_shape().as_list()
        except ValueError:
            _, features_nr = layers[layer_name].get_shape().as_list()
            length = features_nr

        denominator = 4 * length ** 4

        target_gram = gram_matrix(layers[layer_name][0], features_nr=features_nr)
        style_gram = gram_matrix(layers[layer_name][2], features_nr=features_nr)

        style_loss += tf.reduce_sum(tf.square(target_gram - style_gram) / denominator)

    for layer_name in content_layers:
        content_loss += tf.reduce_sum(
            tf.square(layers[layer_name][0] - layers[layer_name][1])
        ) / 2

    return style_loss, content_loss, init_song, content_spectrogram, style_spectrogram


def run_transfer(net_name, weights_filename, song_content, song_style,
                 style_mult, content_mult, iters, frames):
    sess = tf.Session
    style_loss, content_loss, \
    init_song, content_spectrogram, style_spectrogram = build_transferer(net_name, weights_filename,
                                                                         sess, frames)

    loss = style_mult * style_loss + content_mult * content_loss

    opt = tf.train.AdamOptimizer(1e-3).minimize(loss, var_list=[init_song])

    feed_dict = {content_spectrogram: song_content, style_spectrogram: song_style}

    for i in range(iters):

        sess.run(opt, feed_dict=feed_dict)

    return sess.run(init_song)