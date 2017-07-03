import tensorflow as tf
import math
import json
import re
import collections
import numpy as np

BATCH_SIZE = 256
VOCAB_SIZE = 20000
EMBED_SIZE = 300
LEARNING_RATE = 0.95
NUM_SAMPLED = 10
SKIP_WINDOW = 2
NUM_TRAIN_STEPS = 50

with tf.name_scope("data"):
    
    '''Define inputs and outputs
    Center and target word is defined by index
    Target word is randomly selected from the radius from center word of size SKIP_WINDOW
    '''
    
    center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
    target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')
   
with tf.device('/cpu:0'):
    with tf.name_scope("embed"):
        '''Embedding matrix where the word vectors will be found in the end
        '''
    
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),\
                                name='embed_matrix')
     
'''Define inference and loss function
'''
with tf.name_scope("loss"):
     # Initialize embedding lookup matrix for center words
     embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
     # Initialize nce_weight matrix of size [VOCAB_SIZE, EMBED_SIZE] with truncated normal distribution
     nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
     stddev = 1.0 / math.sqrt(EMBED_SIZE)), name='nce_weight')
     # Initialize nce_bias
     nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
     # Define loss function to be NCE loss function with nce_weight and bias
     # Compare with tf.nn.sampled_softmax_loss!   
     loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,\
     biases = nce_bias, labels = target_words, inputs = embed, num_sampled = NUM_SAMPLED, num_classes=VOCAB_SIZE), name='loss')
     # Define GradientDescentOptimizer
     optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()
    
def prepare_dataset(filename_dir):
    filename = 'job_description_total_102350.json'
    #print 'Preparing a dataset...'
    data = json.load(open(filename_dir + filename))
    vocab = []
    lines = [];
    for sentence in data:
        try:
            line = " ".join((sentence["description"]).split()[0:100])
        except:
            continue
        clean_line = clean_str(line)
        lines.append(clean_line)
        words = set(clean_line.split())
        for word in words:
            vocab.append(word)
            
    word_dict = collections.Counter(vocab).most_common(VOCAB_SIZE-1)
    word_index = {word_dict[i][0]:i for i in range(len(word_dict))}
    center = np.zeros((len(lines)*100))
    target = np.zeros((len(lines)*100))
    k = 0
    steps = range(-SKIP_WINDOW,SKIP_WINDOW)
    del steps[SKIP_WINDOW]

    for i in range(2):
        for row in lines:
            line = row.split()
            for i in range(1,len(line)):
                if line[i] not in word_index:
                    continue;
                pos = np.random.choice(steps)
                try:
                    center[k] = int(word_index[line[i]])
                    target[k] = int(word_index[line[i+pos]])
                    k+=1
                except Exception as e:
                            pass
    #print 'Number of training data points: ', k;
    return center[0:k], target[0:k]

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     average_loss = 0.0
     filename_dir = './'
     center,target = prepare_dataset(filename_dir)
     target = np.reshape(target,(-1,1))
     n_samples = len(center)
     for index in xrange(NUM_TRAIN_STEPS):
         for i in range(n_samples/BATCH_SIZE-1):
             # Initialize session by feeding center_words and target_words of size BATCH_SIZE
             # Accumulate loss_batch
             loss_batch, _ = sess.run([loss, optimizer],
             feed_dict = {center_words: center[BATCH_SIZE*i:BATCH_SIZE*(i+1)], target_words: target[BATCH_SIZE*i:BATCH_SIZE*(i+1)]})
             average_loss+= loss_batch
         if (index + 1) % 10 == 0:
             print('Average loss at step {}: {:5.1f}'.format(index + 1,\
            average_loss / ((index + 1)*n_samples/BATCH_SIZE-1)))

# Perform tsne visualization for the most frequent words in the vocabulary using embedding matrix
'''
Save resulting embedding matrix
Pick some indices of words from from word_index and prepare the matrix X with data
X = embed[indices]
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()
'''
