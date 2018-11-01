import nltk
from nltk.corpus import brown
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from collections import OrderedDict
from collections import Counter

import numpy as np
import sys

# Step 1: Import the data, which is the tagged and untagged brown corpus
# Step 2: Find the trigram probabilties using the tagged corpus
# Transition probability: P(tag|prev 2 tags)
# Emission probability: P(word|tag, prev word)


initial = list()
tag_word_pair = OrderedDict()
word_tag_pair = OrderedDict()
transition_probability = OrderedDict()
emission_pairs = OrderedDict()
emission_probability = OrderedDict()


alpha = list()
beta = list()
corpus = brown

for (word, tag) in corpus.tagged_words():
	word_frequency = FreqDist(word)
	tag_frequency = FreqDist(tag)

def calculate():
	train_length = int(0.8 * len(corpus.tagged_sents()))
	print (train_length)
	for sentence in corpus.tagged_sents()[:train_length]:
		bigram = ("<b>",)
		
		
		for counter, word_tag in enumerate(sentence):  
			# Calculate tag|word
			if word_tag[1] not in tag_word_pair:
				tag_word_pair[word_tag[1]] = OrderedDict()
			if word_tag[0] not in tag_word_pair[word_tag[1]]:
				tag_word_pair[word_tag[1]][word_tag[0]] = 0
			tag_word_pair[word_tag[1]][word_tag[0]] += 1

			# Calculate word|tag
			if word_tag[0] not in word_tag_pair:
				word_tag_pair[word_tag[0]] = OrderedDict()
			if word_tag[1] not in word_tag_pair[word_tag[0]]:
				word_tag_pair[word_tag[0]][word_tag[1]] = 0
			word_tag_pair[word_tag[0]][word_tag[1]] += 1
				
			# Emission Pairs
			bigram = bigram + (word_tag[0],)
			key = bigram + (word_tag[1],)
			if key not in emission_pairs:
				emission_pairs[key] = 0
			emission_pairs[key] += 1
			bigram = (word_tag[0],)
				
				
def tag_ngrams(n):
	n_dict = {}
	train_length = int(0.8 * len(corpus.tagged_sents()))
	for sentence in corpus.tagged_sents()[:train_length]:
		# Handling end cases first
		if n == 2:
			br1 = ("<b>",sentence[0][1])
			if br1 not in n_dict:
				n_dict[br1] = 0
			n_dict[br1] += 1
		if n == 3:
			tr1 = ("<t>", "<t>", sentence[0][1])
			if tr1 not in n_dict:
				n_dict[tr1] = 0
			n_dict[tr1] += 1
			if len(sentence) > 2:
				tr2 = ("<t>", sentence[0][1], sentence[1][1])
				trl = (sentence[-2][1], sentence[-1][1], "</t>")
				if tr2 not in n_dict:
					n_dict[tr2] = 0
				if trl not in n_dict:
					n_dict[trl] = 0
				n_dict[tr2] += 1
				n_dict[trl] += 1

		tokens = np.asarray(sentence)
		tokens = tokens[:,1].tolist()
		ngrams = [t for t in nltk.ngrams(tokens, n)]
		for ngram in ngrams:
			if ngram not in n_dict:
				n_dict[ngram] = 0
			n_dict[ngram] += 1
	n_dict = sorted(n_dict.items(), key=lambda x: x[1], reverse=True)
	n_dict = dict((k[0],k[1]) for k in n_dict)        
	return n_dict

tag_trigrams = tag_ngrams(3)
tag_bigrams = tag_ngrams(2)

def calculate_emission():
	number = sum(emission_pairs.values())
	vocab = sum(emission_pairs.values())
	lam = 0.5
	for pair in emission_pairs:
		emission_probability[pair] = (emission_pairs[pair] + lam)/(number+ lam*vocab)
	emission_probability['None'] = lam/(number + lam * vocab)
	
	
def calculate_transition():
	number = sum(tag_trigrams.values())
	vocab = len(tag_trigrams.values())
	lam = 0.5
	for trigram in tag_trigrams:
		transition_probability[trigram] = (tag_trigrams[trigram] + lam)/(number+ lam*vocab)
	transition_probability['None'] = lam/(number + lam * vocab)

def forward_calculate(observations):
	alpha = {}
	for counter, word in enumerate(observations):
		alpha[counter] = {}
		possible = get_possible_tags(word)
		if counter == 0:
			for tag in possible:
				emission = check_existence(emission_probability, ("<b>",word, tag))
				init = initial[tag]
				alpha[counter][tag] = emission * init
			
		else:
			previous = [None]*3
			previous[0] = get_possible_tags(observations[counter-1])
			if counter > 1:
				previous[1] = get_possible_tags(observations[counter-2])
			else:
				previous[1] = ["<t>"]
			for tag in possible:
				x_prob = 0
				emission = check_existence(emission_probability, (observations[counter-1],word,tag))
				sum1 = 0
				for prev0 in previous[0]:
					sum2 = 0
					for prev1 in previous[1]:
						sum2 += check_existence(transition_probability,(prev1, prev0, tag))
					sum1 += alpha[counter - 1][prev0] * sum2
				alpha[counter][tag] = emission*sum1
	probability_observations = 0
	for key in alpha[counter]:
		if key == 'None':
			continue
		probability_observations += alpha[counter][key]
	return probability_observations
		
				
def check_existence(dic, tup):
	if tup in dic:
		return dic[tup]
	return dic['None']

def get_possible_tags(word):
	if word in word_tag_pair:
		return word_tag_pair[word]
	return tag_word_pair.keys()
	 
		
def viterbi(observations):
	delta = {}
	psi = {}
	N = len(observations)
	delta[-1] = {}
	delta[-1][("<t>", "<t>")] = 1
	for counter, word in enumerate(observations):
		previous = [None]*3
		if counter == 0:
			previous = ['<b>', ['<t>'], ['<t>']]
		elif counter == 1:
			previous[0] = observations[0]
			previous[1] = get_possible_tags(observations[0])
			previous[2] = ['<t>']
		else:
			previous[0] = observations[counter - 1]
			previous[1] = get_possible_tags(observations[counter - 1])
			previous[2] = get_possible_tags(observations[counter - 2])
		delta[counter] = {}
		psi[counter] = {}
		possible = get_possible_tags(word)
		for tag in possible:
			emission = check_existence(emission_probability, (previous[0],word,tag))
			for prev1 in previous[1]:
				maxpos = -40000
				argmax = ''
				for prev2 in previous[2]:
					transition_prob = check_existence(transition_probability, (prev2, prev1, tag))
					prod = delta[counter - 1][(prev2, prev1)] * transition_prob
					if prod > maxpos:
						maxpos = prod
						argmax = prev2
				delta[counter][(prev1,tag)] = transition_prob * maxpos
				psi[counter][(prev1,tag)] = argmax                
	temp1 = []
	temp2 = []
		
	temp1 = get_possible_tags(observations[N-1])
	if N - 2 == -1:
		temp2 = ['<t>']
	else:
		temp2 = get_possible_tags(observations[N-2])
			
	best_tags = [None]*N
	maxpos = -1
	argmax = ()
	for prev1 in temp1:
		for prev2 in temp2:
			trans_prob = check_existence(transition_probability, (prev2,prev1,'</t>'))
			prod = delta[N - 1][(prev2, prev1)]* trans_prob
			if prod > maxpos:
				maxpos = prod
				argmax = (prev2, prev1)
		best_tags[N-2] = argmax[0]
		best_tags[N-1] = argmax[1]

	for x in range(N - 3, -1, -1):
		best_tags[x] = psi[x+2][(best_tags[x+1], best_tags[x+2])]
	return best_tags
			
def get_bestfit_tags():
	train_length = int(0.8 * len(corpus.tagged_sents()))
	test_data = corpus.sents()[train_length:]
	correct = 0
	total = 0
	for counter, sentence in enumerate(test_data[:500]):
		pred = viterbi(sentence)
		actual = [tag for word, tag in corpus.tagged_sents()[train_length + counter]]
		total += len(actual)
		for c, predicted in enumerate(pred):
			if predicted == actual[c]:
				correct += 1
	print("Accuracy:", correct/total)
	for i in range(len(sentence)):
		print('{:<17} {}'.format(sentence[i],pred[i]))
	 
def print_most_probable_tags():
	print("Most probable tags for words")
	for pos, val in tag_word_pair.items():
		sorted_val = [word[0] for word in sorted(val.items(),key=lambda x : x[1],reverse = True)[:50]]
		print('{:<10} {} \n'.format(pos,sorted_val))
	print('\n')
 


calculate()
# Randomize initial probability
calculate_emission()
calculate_transition()

# Assign random initial probabilities to tags
initial = np.random.dirichlet(np.ones(len(tag_word_pair.keys())),size=1)
initial = initial.flatten()
initial = np.sort(initial)[::-1]
temp = OrderedDict()
for k1, k2 in zip(tag_word_pair.keys(), initial):
	temp[str(k1)] = k2
initial = temp

print(get_bestfit_tags())


train_length = int(0.8 * len(corpus.tagged_sents()))
test_data = corpus.sents()[train_length:]
print("Sentence: ", " ".join(test_data[0]))
print("Probability of Observation Sequence ",forward_calculate(test_data[0]))