# Pytorch-text-classifier
Implementation of text classification in pytorch using CNN/GRU/LSTM.

This is an in-progress implementation. It is fully functional, but  many of the settings are 
currently hard-coded and it needs some serious refactoring. I was learning pytorch through
writing this code, so, in retrospect, there definitely are parts I should have written differently.

Instructions to run the code (to be executed from the project root folder):
1) unpack the demo_data archive, this will cause a ./demo_data folder with some files to appear
   ```bash
   tar xvzf demo_data.tar.gz
   ```
2) create an environment and install all dependencies
   ```bash
   virtualenv -p python3 env
   source ./env/bin/activate
   pip3 install requirements.txt
   ```bash
3) run the model on the demo data (to run much faster on GPU, set the use_GPU variable near the top of the code to True)
   ```bash
   python3 ./model/run_model.py ./demo_data/
   ```
 After the model training and evaluation is finished, something like this should be printed:

```bash
Finished with training.
Loading checkpoint that was best on validation data
Loading test data (balanced) ...
Creating set of test minibatches ...
Starting eval function
Eval function minibatch 100 / 256.
Eval function minibatch 200 / 256.
Finished with eval, time required was (0 min 10 sec) 

F1 score on balanced test set is 0.842

```


Also, in the ./demo_data, a series of checkpoints will appear, as well as two files -- test_balanced_logits.pickle and test_balanced_true.pickle. Which are the model outputs and true outputs on the test set.

Some functionalities supported by the code (albeit for now in a very hacky non user-friendly way):
- training the model with minibatch SGD (Adam is the optimizer)
- training on GPU
- early stopping by periodically observing performance on the dev set, checkpoints are saved each time an improvement on dev is encountered, the test set is labeled by the model from the best checkpoint
- optional gradient clipping
- option of (1) fine-tuning the embeddings with the rest of the model parameters or (2) keeping them fixed throughout training
- using RNN as the encoder to get text representations (LSTM or GRU, optionally bidirectional, number of units in the cell/hidden state is a hyperparameter, for now with 1 layer but this is easily changed in pytorch)
- using CNN as the encoder, for now all filters need to be of the same width, width and number of filters are hyperparams
- fine tuning the classification threshold to be different than the standard 0.5 in order to maximize F1 on an extremely unbalanced test set if the train data was balanced, this functionality is not used in the demo
