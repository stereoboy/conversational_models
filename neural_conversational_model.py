import numpy as np
import tensorflow as tf
import sys
import scipy.misc
import os
import io
import random
import nltk
import json
from datetime import datetime, date, time
import collections
from tqdm import tqdm  # Progress bar

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("state_dim", "1024", "cell state size")
tf.flags.DEFINE_integer("voc_size", "20000", "vocabulary size")
tf.flags.DEFINE_integer("layer_size", "2", "LSTM layer size")
tf.flags.DEFINE_integer("num_samples", "512", "number of samples for sampled_softmax")
#tf.flags.DEFINE_integer("batch_size", "64", "batch_size")
tf.flags.DEFINE_integer("batch_size", "16", "batch_size")
#tf.flags.DEFINE_integer("max_len", "42", "seq max_size before adding <eos>")
tf.flags.DEFINE_integer("max_len", "10", "seq max_size before adding <eos>")
tf.flags.DEFINE_string("directory", "./", "directory for TFRecords")
tf.flags.DEFINE_integer("max_epoch", "10", "maximum iterations for training")
tf.flags.DEFINE_integer("max_itrs", "10000", "maximum iterations for training")
tf.flags.DEFINE_integer("img_size", "500", "sample image size")
tf.flags.DEFINE_string("dicts_file", "cornell_dicts.json", "dictionary file for saving word2id, id2word")
tf.flags.DEFINE_string("convs_file", "cornell_convs_v3.json", "conversation ids file")
tf.flags.DEFINE_string("save_dir", "cm_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_integer("save_itr", "300", "checkpoint interval")
tf.flags.DEFINE_float("learning_rate", "0.5", "Learning rate for Momentum Optimizer")
tf.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("momentum", "0.9", "momentum for Momentum Optimizer")
tf.flags.DEFINE_float("weight_decay", "0.0016", "Learning rate for Momentum Optimizer")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_float("eps", "1e-5", "epsilon for various operation")
tf.flags.DEFINE_boolean("train", True, "train or inference?")

unknown = 0 #word2id["<unknown>"]
pad =     1 #word2id["<pad>"]
go =      2 #word2id["<go>"]
eos =     3 #word2id["<eos>"]

#V1
#_buckets = [(5,5), (12, 12), (12, 22), (22, 12), (22, 22), (32, 32), (FLAGS.max_len, FLAGS.max_len+1)]
#V2
#_buckets = [(5,5), (12, 12),]
#V3
_buckets = [(FLAGS.max_len, FLAGS.max_len + 1),]
_buckets = [(8, 8)]
#_buckets = [(8, 8), (12, 12)]
#
# original source code of read_data() is from https://github.com/Conchylicultor/DeepQA.git
# thanks to Conchylicutor
#
def word_filter(text):
  text = text.replace("*", "")
  text = text.replace("<u>", "")
  text = text.replace("</u>", "")
  text = text.replace("<U>", "")
  text = text.replace("</U>", "")
  text = text.replace("<i>", "")
  text = text.replace("</i>", "")
  text = text.replace("<I>", "")
  text = text.replace("</I>", "")
  text = text.replace("<b>", "")
  text = text.replace("</b>", "")
  text = text.replace("<pre>", "")
  text = text.replace("</pre>", "")
  text = text.replace("<PRE>", "")
  text = text.replace("</PRE>", "")
  text = text.replace("<html>", "")
  text = text.replace("</html>", "")
  text = text.replace("<", "")
  text = text.replace(">", "")
  text = text.replace(u"\u0092", "'")
  text = text.replace(u"\u0096", " ")
  text = text.replace(u"\u0097", " ")
  text = text.replace(u"\u00ed", "'")
  text = text.replace(u"\u0091", "'")
  text = text.replace(u"\u0094", '"')
  text = text.replace("`", "'")
  text = text.replace(u"`", "'")

  return text
def read_data():
  data_dir = './cornell movie-dialogs corpus'
  lines = {}
  _conversations = []
  MOVIE_LINES_FIELDS = ["lineID","characterID","movieID","character","text"]
  MOVIE_CONVERSATIONS_FIELDS = ["character1ID","character2ID","movieID","utteranceIDs"]

  filepath = os.path.join(data_dir, 'movie_lines.txt')
  with io.open(filepath, mode='r', encoding='iso-8859-1') as f:
    for line in f:
      values = word_filter(line).split(" +++$+++ ")

      # Extract fields
      lineObj = {}
      for i, field in enumerate(MOVIE_LINES_FIELDS):
        lineObj[field] = values[i]

      lines[lineObj['lineID']] = lineObj

  filepath = os.path.join(data_dir, 'movie_conversations.txt')
  with io.open(filepath, mode='r', encoding='iso-8859-1') as f:
    for line in f:
      values = line.split(" +++$+++ ")

      # Extract fields
      convObj = {}
      for i, field in enumerate(MOVIE_CONVERSATIONS_FIELDS):
        convObj[field] = values[i]

      # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
      lineIds = eval(convObj["utteranceIDs"])

      # Reassemble lines
      convObj["lines"] = []
      for lineId in lineIds:
        convObj["lines"].append(lines[lineId])

      _conversations.append(convObj)

  for key, value in _conversations[0].items():
    print key, value

  return _conversations


def build_dict(_conversations):
  dictionary_path = os.path.join("./", FLAGS.dicts_file)
  if os.path.exists(dictionary_path):
    print "load", dictionary_path
    with open(dictionary_path) as f:
      data = json.load(f)
      word2id = data['word2id']
      id2word = {}
      for key, value in data['id2word'].items():
        id2word[int(key)] = value
  else:
    print "create dictionary.json"
    words = []
    for convs in tqdm(_conversations, desc="split text..."):
      for conv in convs['lines']:
        temp = conv['text']
        sentences = nltk.sent_tokenize(temp)
        for s in sentences:
          for w in nltk.word_tokenize(s):
            words.append(w.lower())
    count = []
    base = [
        ['<unknown>', -1],
        ['<pad>', -1],
        ['<go>', -1],
        ['<eos>', -1]
        ]
    count.extend(collections.Counter(words).most_common(FLAGS.voc_size - len(base)))
    count = base + count
    word2id = dict()
    for word, _ in tqdm(count, desc="build dictionary..."):
      word2id[word] = len(word2id)

    assert len(word2id) <= FLAGS.voc_size, "Too small Corpus!"

    id2word = dict(zip(word2id.values(), word2id.keys()))

    print word2id.values()[:10]
    print word2id.keys()[:10]
    with open(dictionary_path, "wb+") as f:
      data = {'word2id':word2id, 'id2word':id2word}
      json.dump(data, f, indent=2)
  return word2id, id2word

def convert_enc(word2id, line):
  words = []
  keys = word2id.keys()
  sentencesToken = nltk.sent_tokenize(line)

  #print "sentencesToken:", sentencesToken
  for i in xrange(len(sentencesToken)):
    i = len(sentencesToken)-1 - i

    tokens = nltk.word_tokenize(sentencesToken[i])

    #print "tokens:", tokens
    # If the total length is not too big, we still can add one more sentence
    if len(words) + len(tokens) <= FLAGS.max_len:
      tempWords = []
      for token in tokens:
        _id = word2id[token.lower()] if token.lower() in keys else unknown
        tempWords.append(_id)  # Create the vocabulary and the training sentences
      words = tempWords + words
    else:
      break  # We reach the max length already

  return words

def convert_dec(word2id, line):
  words = []
  keys = word2id.keys()
  sentencesToken = nltk.sent_tokenize(line)

  #print "sentencesToken:", sentencesToken
  for i in xrange(len(sentencesToken)):

    tokens = nltk.word_tokenize(sentencesToken[i])

    #print "tokens:", tokens
    if len(words) + len(tokens) <= FLAGS.max_len:
      tempWords = []
      for token in tokens:
        _id = word2id[token.lower()] if token.lower() in keys else unknown
        tempWords.append(_id)  # Create the vocabulary and the training sentences
      words = words + tempWords
    else:
      break  # We reach the max length already

  return words

def recover_sentence(id2word, ids):
  words = []
  for i in ids:
    words.append(id2word[i])
  return words

def build_conversations(word2id, id2word, _conversations):
  conversation_path = os.path.join("./", FLAGS.convs_file)
  if os.path.exists(conversation_path):
    print "load", conversation_path
    with open(conversation_path) as f:
      data = json.load(f)
      conversations = data['conversations']

  else:
    conversations = []
    for convs in tqdm(_conversations, desc="convert raw_data into data..."):

      # Iterate over all the lines of the conversation
      for i in xrange(len(convs["lines"]) - 1):  # We ignore the last line (no answer for it)
#      print "---------------------------------------------"
        _encode = convs["lines"][i]
        _decode = convs["lines"][i+1]

        encode = convert_enc(word2id, _encode["text"])
        decode = convert_dec(word2id, _decode["text"])

        if encode and decode:  # Filter wrong samples (if one of the list is empty)
          conversations.append((encode, decode))
#        print _encode["text"]
#        print "encode:", " ".join(recover_sentence(id2word, encode))
#        print _decode["text"]
#        print "decode:", " ".join(recover_sentence(id2word, decode))

    with open(conversation_path, "wb+") as f:
      data = {'conversations': conversations}
      json.dump(data, f, indent=2)

  return conversations

def create_batch(conversations, buckets):
  _batches = [[] for _ in buckets]
  for (encode, decode) in conversations:
    decode =  decode + [eos]
    for bucket_id, (encoder_size, decoder_size) in enumerate(buckets):
      if len(encode) <= encoder_size and len(decode) <= decoder_size:
        _batches[bucket_id].append((encode, decode))

  for bucket_id, (encoder_size, decoder_size) in enumerate(buckets):
    print "[", bucket_id, "](", encoder_size, "x", decoder_size, "):", len(_batches[bucket_id])

  return _batches

def shuffle(_batches):
  batches = []

  for bucket_id, sub in enumerate(_batches):
    random.shuffle(sub)
    for i in xrange(len(sub)//FLAGS.batch_size):
      batches.append((bucket_id, sub[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]))

  random.shuffle(batches)
  return batches

def parse_batch(batch, bucket_size):
  bucket_id = batch[0]
  batch_ = batch[1]
  encoder_size, decoder_size = bucket_size
  encoder_inputs, decoder_inputs = [], []

  #for i, b in enumerate(batch_):
  #  print "[", i, "]", b

  # Get a random batch of encoder and decoder inputs from data,
  # pad them if needed, reverse encoder inputs and add GO to decoder.
  for i in xrange(FLAGS.batch_size):
    encoder_input, decoder_input = batch_[i]

    # Encoder inputs are padded and then reversed.
    encoder_pad = [pad] * (encoder_size - len(encoder_input))
    encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

    # since <go>, len(decoder_inputs) = 1 + decoder_size 
    decoder_pad_size = decoder_size - len(decoder_input)
    decoder_inputs.append([go] + decoder_input +
                          [pad] * decoder_pad_size)

  # Now we create batch-major vectors from the data selected above.
  batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

  # Batch encoder inputs are just re-indexed encoder_inputs.
  for length_idx in xrange(len(encoder_inputs[0])):
    batch_encoder_inputs.append(
        np.array([encoder_inputs[batch_idx][length_idx]
                  for batch_idx in xrange(FLAGS.batch_size)], dtype=np.int32))

  # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
  for length_idx in xrange(len(decoder_inputs[0])):
    batch_decoder_inputs.append(
        np.array([decoder_inputs[batch_idx][length_idx] 
                  for batch_idx in xrange(FLAGS.batch_size)], dtype=np.int32))

    # Create target_weights to be 0 for targets that are padding.
    batch_weight = np.ones(FLAGS.batch_size, dtype=np.float32)
    for batch_idx in xrange(FLAGS.batch_size):
      # We set weight to 0 if the corresponding target is a PAD symbol.
      # The corresponding target is decoder_input shifted by 1 forward.
      if length_idx < decoder_size:
        target = decoder_inputs[batch_idx][length_idx + 1]
#      if length_idx == decoder_size:
#        batch_weight[batch_idx] = 0.0
      if target == pad:
        batch_weight[batch_idx] = 0.0
    batch_weights.append(batch_weight)

  return batch_encoder_inputs, batch_decoder_inputs, batch_weights

def get_input(buckets, dtype=np.float32):
  print "setup input..."
  # Feeds for inputs.
  x = []
  y = []
  t_w = []
  for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
    x.append(tf.placeholder(tf.int32, shape=[None],	name="encoder{0}".format(i)))
  for i in xrange(buckets[-1][1] + 1): # +1 is for setting up targets below
    y.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
    t_w.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))
  # Our targets are decoder inputs shifted by one.
  t = [y[i + 1] for i in xrange(len(y) - 1)]

  #
  #             t: y[1] y[2] ... y[max_size]     | .
  #             y: y[0] y[1] ... y[max_size - 1] | y[max_size]
  #             w: w[0] w[1] ... w[max_size - 1] | w[max_size]
  return x, y, t, t_w

def set_feed((x, x_inputs), (y, y_inputs), (t_w, t_w_inputs), bucket_size):
  encoder_size, decoder_size = bucket_size

  # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
  input_feed = {}
  for l in xrange(len(x_inputs)):
    input_feed[x[l].name] = x_inputs[l]
  for l in xrange(len(y_inputs)):
    input_feed[y[l].name] = y_inputs[l]
    input_feed[t_w[l].name] = t_w_inputs[l]

  # Since our targets are decoder inputs shifted by one, we need one more.
  #last_target = y[decoder_size].name
  #input_feed[last_target] = np.zeros([FLAGS.batch_size], dtype=np.int32)

  return input_feed

def model(x, y, t, t_w, buckets, train=True):
  print "setup model..."

  w = tf.get_variable("proj_w", [FLAGS.state_dim, FLAGS.voc_size], dtype=tf.float32)
  w_t = tf.transpose(w)
  b = tf.get_variable("proj_b", [FLAGS.voc_size], dtype=tf.float32)
  output_projection = (w, b)

  single_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.state_dim)
  cell = tf.nn.rnn_cell.MultiRNNCell([single_cell]*FLAGS.layer_size)

  def sampled_loss(inputs, labels):
    labels = tf.reshape(labels, [-1, 1])
    # We need to compute the sampled_softmax_loss using 32bit floats to
    # avoid numerical instabilities.
    local_inputs = tf.cast(inputs, tf.float32)
    return tf.cast(
        tf.nn.sampled_softmax_loss(w_t, b, local_inputs, labels,
          FLAGS.num_samples, FLAGS.voc_size),
        dtype=tf.float32)
  softmax_loss_function = sampled_loss

  # The seq2seq function: we use embedding for the input and attention.
  def seq2seq_f(x, y, do_decode):
    return tf.nn.seq2seq.embedding_attention_seq2seq(
        x,
        y,
        cell,
        FLAGS.voc_size,
        FLAGS.voc_size,
        FLAGS.state_dim,
        output_projection=output_projection,
        feed_previous=do_decode,
        dtype=tf.float32
        )

  # Training outputs and losses.
  if not train:
    do_decode = True
  else:
    do_decode = False

  _outputs, losses = tf.nn.seq2seq.model_with_buckets(
      x, y, t,
      t_w, buckets,
      lambda x, y: seq2seq_f(x, y, True),
      softmax_loss_function=softmax_loss_function)
  # only for inference step
  # If we use output projection, we need to project outputs for decoding.
  outputs = []
  for b_id in xrange(len(buckets)):
    outputs.append([tf.matmul(_output, w) + b for _output in _outputs[b_id]])

  return outputs, losses

def get_opt(losses, buckets, dtype=tf.float32):
  print "setup optimizer..."
  # Gradients and SGD update operation for training the model.
  learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False, dtype=dtype)
  learning_rate_decay_op = learning_rate.assign(learning_rate*FLAGS.learning_rate_decay_factor)
  global_step = tf.Variable(0, trainable=False)

  params = tf.trainable_variables()
  gradient_norms = []
  updates = []
  opt = tf.train.GradientDescentOptimizer(learning_rate)
  for b in xrange(len(buckets)):
    gradients = tf.gradients(losses[b], params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
    gradient_norms.append(norm)
    updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step))

  return updates, gradient_norms, learning_rate_decay_op

def print_now(start):
  current = datetime.now()
  print "\telapsed:", current - start

def process(train=True):
  start = datetime.now()
  print "Start: ",  start.strftime("%Y-%m-%d_%H-%M-%S")

  _conversations = read_data()
  word2id, id2word = build_dict(_conversations)
  print word2id.items()[:10]
  print id2word.items()[:10]
  conversations = build_conversations(word2id, id2word, _conversations)
  print_now(start)

  _batches = create_batch(conversations, _buckets)
  print_now(start)

  x, y, t, t_w = get_input(_buckets)
  print_now(start)

  outputs, losses = model(x, y, t, t_w, _buckets, True)
  print_now(start)

  for output in outputs:
    print output
  outtexts = [[] for _ in xrange(len(_buckets))]
  for b_id, bucket in enumerate(_buckets):
    print bucket
    _, decoder_size = bucket
    for l in xrange(decoder_size):  # Output logits.
      outtexts[b_id].append(tf.argmax(outputs[b_id][l][0], 0))
#
#  #infer_output = model(x, y, False)
#
  updates, gradient_norms, learning_rate_decay_op = get_opt(losses, _buckets)
  print_now(start)

  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  for i, var in enumerate(var_list):
    print "[", i, "]", var
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  with tf.Session() as sess:
    # Initialize the variables (the trained variables and the
    sess.run(init_op)

    previous_losses = []
    loss = 0.0

    # Start input enqueue threads.
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(FLAGS.save_dir)
    print "checkpoint: %s" % checkpoint
    if checkpoint:
      print "Restoring from checkpoint", checkpoint
      saver.restore(sess, checkpoint)
    else:
      print "Couldn't find checkpoint to restore from. Starting over."
      dt = datetime.now()
      filename = "checkpoint" + dt.strftime("%Y-%m-%d_%H-%M-%S")
      checkpoint = os.path.join(FLAGS.save_dir, filename)

    for epoch in xrange(FLAGS.max_epoch):
      print "#####################################################################"
      batches = shuffle(_batches)
      batch_len = len(batches)
      for itr in xrange(batch_len):
        print "==================================================================="
        print "[", epoch, "]", "%d/%d"%(itr, batch_len)

        batch = batches[itr]
        bucket_id = batch[0]
        bucket_size = _buckets[bucket_id]
        encoder_size, decoder_size = bucket_size
        x_inputs, y_inputs, t_w_inputs = parse_batch(batch, bucket_size)
        feed_dict = set_feed((x, x_inputs), (y, y_inputs), (t_w, t_w_inputs), bucket_size)

        (q_ids, a_ids) = batch[1][0]
        q = " ".join(recover_sentence(id2word, q_ids))
        a = " ".join(recover_sentence(id2word, a_ids))
        # Output feed: depends on whether we do a backward step or not.
        if train:
          output_feed = [updates[bucket_id],  # Update Op that does SGD.
              gradient_norms[bucket_id],  # Gradient norm.
              losses[bucket_id]]  # Loss for this batch.
          #output_feed.append(outtexts[bucket_id])

          #_, _, loss = sess.run(output_feed, feed_dict)
          outs = sess.run(output_feed, feed_dict)
          print "loss:", outs[2]
          ids = sess.run(outtexts[bucket_id], feed_dict)
          final = " ".join(recover_sentence(id2word, ids))
          print "Q:", q
          print "A:", a
          print "G:", final
        else:
          _, _, loss = sess.run(output_feed, feed_dict)
          print "loss:", loss
          ids = sess.run(outtexts[bucket_id], feed_dict)
          final = " ".join(recover_sentence(id2word, ids))
          print "Q:", q
          print "A:", a
          print "G:", final

#        _, loss_val = sess.run([opt, loss], feed_dict=feed_dict)
#        accuracy_val = sess.run([accuracy], feed_dict=feed_dict)
#        print "\tloss:", loss_val
#        print "\taccuracy:", accuracy_val

        #images_value, labels_value = sess.run([x, y], feed_dict=feed_dict)
#        indexed_out_val = sess.run(indexed_out, feed_dict=feed_dict)

        current = datetime.now()
        print "\telapsed:", current - start

#        if itr > 1 and itr %10 == 0:
#          # Decrease learning rate if no improvement was seen over last 3 times.
#          if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
#            sess.run(learning_rate_decay_op)
#          previous_losses.append(loss)


        if itr > 1 and itr % FLAGS.save_itr == 0:
          print "#######################################################"
          saver.save(sess, checkpoint)

def infer():

  return None

def main(args):
  if FLAGS.train == True:
    print "########################################################"
    print "train()"
    process()
  else:
    print "########################################################"
    print "infer()"
    process(train=False)


if __name__ == "__main__":
  tf.app.run()
