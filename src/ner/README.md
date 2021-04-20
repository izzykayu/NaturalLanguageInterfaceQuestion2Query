Named Entity recognition using RNN
Steps:
1. Use pre-trained word embeddings and fine tune them on MEDLINE. Save the new word embeddings. [code using gensim's word2vec]
2. Load the above word embeddings and train word level bi-LSTM. Output of bi-LSTM is fed into simple softmax layer. [code using pytorch]
	Notes: “BIOES” (B-beginning of an entity, I-insider an entity, O-outsider an entity, E-end of an entity, S-a single-token entity) to represent entities.
	Blocker: Need labeled training data with sentences and entity types like drug, disease, symptoms. 
3. Measure accuracy/ F-1 score of bi-LSTM with simple softmax as output layer 
4. Replace simple softmax to include CRF as output. [paper uses CRFsuite package for coding]
	Notes: Softmax layer doesnt consider interactions between successive labels. For example, if a <B-beginning of an entity> tag is predicted then probability that next tag is <I-insider an entity> is quite high. This is captured using CRF and a simple softmax cannot do this.
	Blocker: No experience with CRFsuite.
		 Not sure how to backpropogate loss to bi-LSTM in this case
		 
		 
Progress:
Coded the Bi LSTM with CRF Part
Accounted for the pre-trained word embeddings in the code

Next Steps:
Read in the training data
Evaluate the accuracy
			 
