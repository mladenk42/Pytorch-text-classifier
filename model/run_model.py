import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnnutils
from sklearn.metrics import f1_score
import pickle
import time
import sys
import math

outprefix = sys.argv[1]

torch.manual_seed(42)

lstm_hidden_size = 50
bidirectional = True
num_classes = 2
dropout = 0.0 # appears in pytorch this is meant to apply dropout after layers (all but the last one)  of a multi layer lstm, we only have one layer so let's set it to 0


use_GPU = False

cuda_device = torch.device("cuda") if use_GPU else None
 
# Load the data from json

print("Loading (balanced) train data ...")
train_data_X = pickle.load(open(outprefix + "train_data_balanced.pickle","rb"))
train_data_y = torch.load(outprefix + "train_labels_balanced.pt")

print("Loading (balanced) dev data ...")
dev_data_X = pickle.load(open(outprefix + "dev_data_balanced.pickle","rb"))
dev_data_y = torch.load(outprefix + "dev_labels_balanced.pt")



# construct minibatches
print("Rearranging data into minibatches ...")

minibatch_size = 64

def chunks(l, n):
  return [l[i:i + n] for i in range(0, len(l), n)]  
  #for i in xrange(0, len(l), n):
  #    yield l[i:i + n]

def create_minibatches(data_X, data_y, minibatch_size, cuda_dev):
  for idx_list in chunks(range(len(data_X)), minibatch_size):
    data_X_idx = [data_X[index] for index in idx_list]
    data_X_idx = [x[0:100] for x in data_X_idx] # cutoff comments to only the first 100 tokens
    data_y_idx = [int(data_y[index]) for index in idx_list]
    Xy = zip(data_X_idx, data_y_idx) # zip to permute the labels as well when sorting by length
    Xy = sorted(Xy, key = lambda x: len(x[0]), reverse = True)
    minibatch_X, minibatch_y = [list(x) for x in zip(*Xy)] # undo the zip thing to get separate variables for data and labels
    minibatch_lengths = torch.tensor([len(x) for x in minibatch_X])
    minibatch_X = rnnutils.pad_sequence(minibatch_X, padding_value = 0) 
    minibatch_y = torch.tensor(minibatch_y)
    if cuda_dev is not None:
      minibatch_lengths = minibatch_lengths.to(cuda_dev)
      minibatch_X = minibatch_X.to(cuda_dev)
      minibatch_y = minibatch_y.to(cuda_dev)

    yield((minibatch_X, minibatch_y, minibatch_lengths))   

print("Creating set of train minibatches ...")
train_minibatches = list(create_minibatches(train_data_X, train_data_y, minibatch_size, cuda_device))
print("Created " + str(len(train_minibatches)) + " minibatches.")


print("Creating set of dev minibatches ...")
dev_minibatches = list(create_minibatches(dev_data_X, dev_data_y, minibatch_size, cuda_device))
print("Created " + str(len(dev_minibatches)) + " minibatches.")
 
print("Loading the vocabulary dict...")
vocab = pickle.load(open(outprefix + "vocab.pickle","rb"))
vocab_size = len(vocab)


embedding_weights = torch.load(outprefix + "embedding_weights.pt") # loads the "embedding_weights" variable which is the matrix of embeddings 
embedding_size = embedding_weights.shape[1]


print("embedding size is %d" % (embedding_size))


# models and training ...
class myLSTMClassifier(nn.Module):
  def __init__(self, vocab_size, embedding_size, lstm_hidden_size, num_classes, bidirectional = False, pretrained_embeddings = None, dropout = 0, freeze_embeddings = True):   
    super(myLSTMClassifier, self).__init__()

    self.model_type = "LSTM"

    self.emb_size = embedding_size
    self.hid_size = lstm_hidden_size
    self.n_classes = num_classes
    self.bidir = bidirectional
    self.vocab_size = vocab_size
    self.dropout = dropout
    self.embedding_layer = nn.Embedding(self.vocab_size, self.emb_size, padding_idx = 0)
    if pretrained_embeddings is not None:
      self.embedding_layer.weight.data.copy_(pretrained_embeddings)
    if freeze_embeddings:
      self.embedding_layer.weight.requires_grad = False

    if self.model_type == "LSTM":
      self.rnn_layer = nn.LSTM(
            input_size=self.emb_size,
            hidden_size=self.hid_size,
            num_layers=1,
            bidirectional=self.bidir,
            dropout=self.dropout
      )
    elif self.model_type == "GRU":
      self.rnn_layer = nn.GRU(
            input_size=self.emb_size,
            hidden_size=self.hid_size,
            num_layers=1,
            bidirectional=self.bidir,
            dropout=self.dropout
      )
    elif self.model_type == "CNN":
      self.kernel_height = 3
      self.kernel_width = self.emb_size
      self.num_kernels = 100
      self.conv_layer = torch.nn.Conv2d(1,self.num_kernels, (self.kernel_width, self.kernel_height))
      self.maxpool_layer = torch.nn.AdaptiveMaxPool2d((1,1))      
    else:
      raise(Exception("Something is wrong."))
   
    if self.model_type == "LSTM" or self.model_type == "GRU":
      encoder_dim = 2 * self.hid_size if self.bidir else self.hid_size
    elif self.model_type == "CNN":
      encoder_dim = self.num_kernels
    self.decoder = nn.Linear(encoder_dim, num_classes)



  def forward(self, input_batch, seq_lengths):
    # embedding layer
    expanded = self.embedding_layer(input_batch)

    if self.model_type =="LSTM" or self.model_type == "GRU":
      expanded = rnnutils.pack_padded_sequence(expanded, seq_lengths)
      rnn_output, hidden = self.rnn_layer(expanded)
      unpacked = rnnutils.pad_packed_sequence(rnn_output)[0]
    
      #final_timesteps = [x-1 for x in lengths]
      final_rnn_output = hidden
      if self.model_type == "LSTM":
        final_rnn_output = final_rnn_output[0] # final hidden state because LSTM outputs both the hidden and the cell states

      if self.bidir:
        final_rnn_output = torch.cat([final_rnn_output[0], final_rnn_output[1]], dim=1) # cat together the final state for the two passes
      else:
        final_rnn_output = final_rnn_output[0]
      encoder_output = final_rnn_output
      
    elif self.model_type == "CNN":
      max_timesteps = input_batch.shape[0]
      num_examples = input_batch.shape[1]
      expanded_BTF = expanded.permute(1,0,2) # TIMESTEPS x BATCHES x FEATS --> BATCHES x TIMESTEPS x FEATS
      expanded_B1TF = expanded_BTF.unsqueeze_(1) # BATCHES x TIMESTEPS x FEATS --> BATCHES x 1 x TIMESTEPS x FEATS (like an image with 1 channel instead of the usual 3)
      conv_output = self.conv_layer(expanded_B1TF)
      maxpool_output = self.maxpool_layer(conv_output)
      encoder_output = maxpool_output.view(num_examples, self.num_kernels)
    else:
      raise(Exception("Something is very very wrong."))

    # linear layer + softmax
    decoder_output = self.decoder(encoder_output) # the log softmax is applied to this later when calculating the loss
    #decoder_softmax_output = F.softmax(decoder_output, dim=1)
    return(decoder_output)



clf = myLSTMClassifier(vocab_size, embedding_size, lstm_hidden_size, num_classes, bidirectional = bidirectional, pretrained_embeddings = embedding_weights, dropout = dropout, freeze_embeddings = False)
if use_GPU:
  clf = clf.to(cuda_device)



# training algorithm
def train_model(model, train_batches, val_batches, loss_function, learning_rate, num_epochs, validation_freq, minibatch_print_freq):
  params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters())) # this will exclude the embeddings from optimization if their requires_grad was set to false
  optimizer = torch.optim.Adam(params_to_optimize, learning_rate, amsgrad = True)
  best_val_F1 = 0.0
  best_checkpoint_filename = ""
  for epoch in range(num_epochs):
    epoch_start = time.time() 
    total_loss = 0
    model.train() # puts the model into "train" mode which affects some types of layers like dropout
    # its important the above line is inside the epochs loop since the validation step after some epochs can change the mode of the model
    current_minibatch = 0
    for train_X, train_y, text_lengths in train_batches:
      current_minibatch += 1
      model.zero_grad()
      # forward pass
      logits = model(train_X, text_lengths)
 
      # calc loss
      minibatch_loss = loss_function(logits, train_y)
      total_loss += minibatch_loss
    
      #backward pass
      minibatch_loss.backward()
      torch.nn.utils.clip_grad_norm_(params_to_optimize, 5) 
      optimizer.step()
      
      if current_minibatch % minibatch_print_freq == 0:
        print("Epoch %d, minibatch %d / %d, minibatch loss is %.3f." % (epoch, current_minibatch, len(train_batches), minibatch_loss))
    epoch_finish = time.time()
    duration = epoch_finish - epoch_start
    print("Finished with epoch %d, total loss is now %.3f. (%d min %d sec) " % (epoch, total_loss, int(duration / 60), int(duration % 60)))
    if epoch % validation_freq == 0: # check performance on validation data
      print("Calculating validation performance.")
      val_loss, val_F1 = eval_model(model, val_batches, loss_function, minibatch_print_freq, len(val_batches))

      print("-----------------------")
      print("Validation loss is %.3f and validation F1 is %.3f" % (val_loss, val_F1))
      print("-----------------------")

      if val_F1 > best_val_F1:
        best_val_F1 = val_F1
        # save a checkpoint
        best_checkpoint_filename = outprefix + "checkpoint-epoch-%05d-val_F1-%.3f.pt" % (epoch, val_F1)
        torch.save({
          "epoch" : epoch,
          "model_state_dict" : model.state_dict(),
          "optimizer_state_dict" : optimizer.state_dict(),
          "train_loss" : total_loss,
          "val_loss" : val_loss,
          "val_F1" : val_F1 
        }, best_checkpoint_filename)        
  print("Finished with training.")
  print("Loading checkpoint that was best on validation data")
  checkpoint = torch.load(best_checkpoint_filename)
  model.load_state_dict(checkpoint["model_state_dict"])
 
# the reason why the number of minibatches is passed instead of just inffered from "eval batches" is that inferring it
# would enumerate the enumerator and that might copy more data to the gpu than it has memory to handle, for the very big
# test set we copy minibatch by minibatch to the gpu (there is probably a better way but the deadline is near)



def calcMeasures(total_logits, total_true, threshold):
    total_preds = []
    sm_func = torch.nn.Softmax(dim=1)
    for minibatch_logits in total_logits:
      sm_res = sm_func(minibatch_logits)
      minibatch_preds = sm_res[:,1] > threshold
      total_preds.append(minibatch_preds)
    total_preds = torch.cat(total_preds)
    total_true = torch.cat(total_true)
    
    F1 = f1_score(total_true.cpu(), total_preds.cpu(), pos_label = 1)
    return(F1)

def apply_model(model, eval_batches, loss_function, minibatch_freq, num_batches): # eval model, both in terms of loss and F1 score 
  model.eval() # puts the model into eval mode (for similar reasons as with model.train()  above)
  with torch.no_grad(): # makes computation more efficient and we don't need gradients here anyway
    total_logits= []
    total_true = []
    total_loss = 0
    current_minibatch = 0
    print("Starting eval function")
    eval_start = time.time()
    for eval_X, eval_y, text_lengths in eval_batches:
      current_minibatch += 1
      minibatch_logits = model(eval_X, text_lengths)
      minibatch_loss = loss_function(minibatch_logits, eval_y)
      total_loss += minibatch_loss
      total_logits.append(minibatch_logits)
      total_true.append(eval_y)
      if current_minibatch % minibatch_freq == 0:
        print("Eval function minibatch %d / %d." % (current_minibatch, num_batches))
    eval_finish = time.time()
    duration = eval_finish - eval_start
    print("Finished with eval, time required was (%d min %d sec) " % (int(duration / 60), int(duration % 60)))
 
    return(total_loss / float(len(total_true)), total_logits, total_true)

def eval_model(model, eval_batches, loss_function, minibatch_freq, num_batches): # eval model, both in terms of loss and F1 score 
    avg_loss, total_logits, total_true = apply_model(model, eval_batches, loss_function, minibatch_freq, num_batches)
    F1 = calcMeasures(total_logits, total_true, 0.5)
    return((avg_loss, F1))


CE_loss = nn.CrossEntropyLoss()
learning_rate = 0.001
n_epochs = 5
validation_f = 1
minibatch_print_f = 100



train_model(clf, train_minibatches, dev_minibatches, CE_loss, learning_rate, n_epochs, validation_f, minibatch_print_f)



# apply model to test set

fine_tune_threshold = True
process_test_set = True

if process_test_set:
 
  print("Loading test data (balanced) ...")
  test_data_X = pickle.load(open(outprefix + "test_data_balanced.pickle","rb"))
  test_data_y = torch.load(outprefix + "test_labels_balanced.pt")
  
  print("Creating set of test minibatches ...")
  test_minibatches = create_minibatches(test_data_X, test_data_y, minibatch_size, cuda_device)
  _, test_balanced_logits, test_balanced_true = apply_model(clf, test_minibatches, CE_loss, minibatch_print_f, math.ceil(len(test_data_X) / minibatch_size))
  test_F1 = calcMeasures(test_balanced_logits, test_balanced_true, 0.5)
  #_, test_F1 = eval_model(clf, test_minibatches, CE_loss, minibatch_print_f, math.ceil(len(test_data_X) / minibatch_size))
  print("\nF1 score on balanced test set is %.3f\n" % (test_F1))
  # save predictions for statistical tests later
  pickle.dump(test_balanced_logits, open( outprefix + "test_balanced_logits.pickle", "wb" ))
  pickle.dump(test_balanced_true, open( outprefix +  "test_balanced_true.pickle", "wb" ))

  print("Done for now")
  exit()

  if fine_tune_threshold:
    print("Fine tuning decision threshold for the unbalanced test data.")
    print("Loading (unbalanced) dev data ...")
    dev_data_X = pickle.load(open(outprefix + "dev_data_unbalanced.pickle","rb"))
    dev_data_y = torch.load(outprefix + "dev_labels_unbalanced.pt")
    dev_minibatches = create_minibatches(dev_data_X, dev_data_y, minibatch_size, cuda_device)
    _,dev_unbalanced_logits, dev_unbalanced_true = apply_model(clf, dev_minibatches, CE_loss, minibatch_print_f, math.ceil(len(dev_data_X) / minibatch_size))    
 
    left_limit = 0.0
    right_limit = 1.0
    current_F1 = calcMeasures(dev_unbalanced_logits, dev_unbalanced_true, 0.5)
    epsilon = 0.000001
    while True:
      print("Current threshold: " + str((left_limit + right_limit) / 2))
      low_F1 = calcMeasures(dev_unbalanced_logits, dev_unbalanced_true, (left_limit + 0.25 * (right_limit - left_limit)))
      high_F1 = calcMeasures(dev_unbalanced_logits, dev_unbalanced_true, (left_limit + 0.75 * (right_limit - left_limit)))
      if low_F1 > high_F1:
        right_limit = (left_limit + right_limit) / 2
      else:
        left_limit = (left_limit + right_limit) / 2
      if (right_limit - left_limit) < epsilon:
        optimized_threshold = (left_limit + right_limit) / 2
        optimized_F1 = calcMeasures(dev_unbalanced_logits, dev_unbalanced_true, optimized_threshold)
        print("Fine tuning complete, best threshold was %.3f giving an F1 score of %.3f" % (optimized_threshold, optimized_F1))
        break
  else:
    optimized_threshold = 0.5
    

   

  print("Loading test data (unbalanced) ...")
  test_data_X = pickle.load(open(outprefix + "test_data_unbalanced.pickle","rb"))
  test_data_y = torch.load(outprefix + "test_labels_unbalanced.pt")
  
  print("Creating set of test minibatches ...")
  test_minibatches = create_minibatches(test_data_X, test_data_y, minibatch_size, cuda_device)
  _, test_unbalanced_logits, test_unbalanced_true = apply_model(clf, test_minibatches, CE_loss, minibatch_print_f, math.ceil(len(test_data_X) / minibatch_size))
  test_F1 = calcMeasures(test_unbalanced_logits, test_unbalanced_true, optimized_threshold)

  print("\nF1 score on unbalanced test set is %.3f\n" % (test_F1))

  pickle.dump(test_unbalanced_logits, open( outprefix + "test_unbalanced_logits.pickle", "wb" ))
  pickle.dump(test_unbalanced_true, open( outprefix +  "test_unbalanced_true.pickle", "wb" ))





