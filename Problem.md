# Problem statement

Implement a HMM using the Brown Corpus (you could import it from nltk). You may use the tagged corpus to calculate the transistion and emission probabilities. 
Note: Consider trigrams (the previous 2 tags) while calculating the transition probabilities, and bigrams (the previous word as well for a given tag) for the emission probabilities. 

Using these computed values, find the probability of the observation sequence given the parameters using Forward, Backward procedure followed by viterbi. For this part of the experiment, you must use the untagged Brown Corpus (consider an 80:20 split). 

Next, given the observation sequence, find the best fit "tag" sequence. 
For each tag, print out the top 50 words (most probable) emitted by it. 

Use random initialisations for the initial state probabilities. Document all your observations into a report.
Note: Before working on the data, make sure to tokenize it.  

BONUS: You will be awarded bonus marks based on the level of innovation (use of extra features) that you bring to the experiment.
