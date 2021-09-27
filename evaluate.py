# ref: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

from torch import optim
from model import *
import os
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

USE_CUDA=True
# set cuda device and seed
if USE_CUDA:
    torch.cuda.set_device(0)
torch.cuda.manual_seed(1)



teacher_forcing_ratio = 0.5

attn_model = 'dot'
hidden_size = 500
n_layers = 2
dropout = 0.1
batch_size = 50


# Configure training/optimization

teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 50000

print_every = 100
evaluate_every = 100

patience = 20

def evaluate(input_batches, input_lengths, target_batches, target_lengths, seq2seq, criterion, max_length=MAX_LENGTH):
    with torch.no_grad():
        decoder_output, loss, _, _ = seq2seq(input_batches, input_lengths, target_batches, target_lengths, batch_size, criterion)

        return loss.data



def main():

    test_lang, test_pairs = prepare_data("data/valid.txt")
    MIN_COUNT = 5

    test_lang.trim(MIN_COUNT)

    keep_pairs = []

    for pair in test_pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in test_lang.word2index:
                keep_input = False
                break

        for word in output_sentence.split(' '):
            if word not in test_lang.word2index:
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from %d pairs to %d, %.4f of total" % (len(test_pairs), len(keep_pairs), len(keep_pairs) / len(test_pairs)))
    test_pairs = keep_pairs

    # Initialize models
    encoder = EncoderRNN(test_lang.n_words, hidden_size, n_layers, dropout=dropout)
    decoder = LuongAttnDecoderRNN(attn_model, hidden_size, test_lang.n_words, n_layers, dropout=dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    seq2seq = Seq2Seq(encoder, decoder, encoder_optimizer, decoder_optimizer)
    seq2seq.load_state_dict(torch.load('checkpoint.pt'))

    if USE_CUDA:
        seq2seq.cuda()



    if USE_CUDA:
        seq2seq.cuda()

    test_input_batches, test_input_lengths, test_target_batches, test_target_lengths = random_batch(batch_size, test_pairs, test_lang)
    with torch.no_grad():
        decoder_output, loss, _, _ = seq2seq(input_batches, input_lengths, target_batches, target_lengths, batch_size, criterion)

    results = []
    for t in range(batch_size):
        for di in range(max_length):
            decoded_words = []
            # Choose top word from output
            topv, topi = decoder_output[t].data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[ni])
        results.append(decoded_words)

    print(results)

main()