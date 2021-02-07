# DeepNLP chatbot v1 

import numpy as np
import tensorflow as tf
import re
import time

lines = open('data/movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('data/movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# lines
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# conversation lines    
conversations_ids = []
for conversation in conversations[:-1]: # The last line in conversation file is empty
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "") # get the last column and exclude the square brackets
    conversations_ids.append(_conversation.split(','))

# Extracting questions and answers    
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        i = i+2

# Clean the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"why's", "why is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"shan't", "shall not", text)
    text = re.sub(r"[-()\"#;:<>{}+=`|.?,]", "", text)
    return text

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))    
    

# Creating a dictionary that maps each work to the number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if(word not in word2count):
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for answer in clean_answers:
    for word in answer.split():
        if(word not in word2count):
            word2count[word] = 1
        else:
            word2count[word] += 1
            
# Creating two dictionaries that maps the question words and the answer words to a unique integer
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if(count >= threshold):
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if(count >= threshold):
        answerswords2int[word] = word_number
        word_number += 1

# Adding the last tokens to these two dictionaries        
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
    answerswords2int[token] = len(answerswords2int) + 1
    
# Creating the inverse dictionary of the answerswords2int dictionary
answersint2word = {w_i: w for w, w_i in answerswords2int.items()}

# Adding the end of the string to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translate all questions and answers into integers
# and replace all the words that were filtered out with <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)


answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

# Sorting questions and ansewrs by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])



##################### BUILDING THE SEQ2SEQ MODEL #####################
            
# Creating placeholders for the inputs and the targets            
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob
            
# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], words2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat( [left_side, right_side], 1)
    return preprocessed_targets

# Creating the encoder RNN Layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, )




















    