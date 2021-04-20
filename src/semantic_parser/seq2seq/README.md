Input: We are running on entity anonymized English-SQL data

Data pre-processing:

(DONE)1. All words that occur only once replaced by UNK (For target??DOUBT)

(DONE)2. Generate Vocabulary, word2index, index2word (template types and number too??) (Should number have their own type??)

(DONE)3. Add special symbols to #2 for <start>(start of sentence), </start>(end of sentence), <padding>(is padding required? No)

(DONE)4. Filter pretrained wordembedding to include only the words in new vocabulary.

(DONE)5. Reorder word embedding based on word2index (what will be the word embedding for start, end and padding?random init)


Model:

(Done)6. Bi-LSTM Encoder(what are the dimensions??)

(Done)7. Bi-LSTM Decoder with attention(what are the dimensions??)

8. Cross validation and minibatch size = 100 

(Done)9. Adam with lr=0.001 for 70 epochs

10. Hyper parameter tuning and early stopping on dev set of cross validation


Prediction:

11. Beam search for decoding(beam size=5)(why beam search?)(how to do beam search in pytorch?)

12. Adding entity to anonymized SQL

13. Executing SQL in python to get results

Notes:
Currently training on geo dataset

Code should be generic enough to run for clinical dataset

Accuracy: How effectively we convert to SQL. Cant be on correct answers.
