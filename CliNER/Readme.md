## i2b2 2012 tags description
1. clinical concepts (problems, tests, and treatments)
2. clinical departments (such as ‘surgery’ or ‘the main floor’)
3. evidentials (ie, events that indicate the source of the information, such as the word ‘complained’ in ‘the patient complained about …’)
4. occurrences (ie, events that happen to the patient, such as ‘admission’, ‘transfer’, and ‘follow-up’).

## model usage
1. chmod +x cliner
2. unzip the model in models file
3. for prediction, run `./cliner predict --txt data/<insert-file-name>.txt --out data/predictions/ --format i2b2 --model models/<insert-model-name>.model`
4. for training, run `./cliner train --txt "data/<insert-training-txt-folder>/" --annotations "data/<insert-training-annotations-folder>/" --format i2b2 --model models/foo.model`


## cliner redone for newest challenge n2c2 track2

1. source activate myenvpython2
2. then model usage
3.
4.
