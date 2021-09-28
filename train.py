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
MAX_LENGTH = 10
def predict(input_batches, decoder_output, lang):

    results = []
    for t in decoder_output:
        decoded_words = []
        for di in t:
            # Choose top word from output

            word = di.data
            topv, topi = word.topk(1)

            ni = topi.item()
            word = lang.index2word[ni]
            decoded_words.append(word)

        results.append(decoded_words)

    inputs = list()
    for t in input_batches:
        words = list()
        for i in t:
            words.append(lang.index2word[i.item()])
        inputs.append(words)


    for input, output in zip(inputs, results):
        input_sentence = ' '.join(input)
        output_sentence = ' '.join(output)

        print('>', input_sentence)
        print('<', output_sentence)
        print('')

def evaluate(input_batches, input_lengths, target_batches, target_lengths, seq2seq, criterion, max_length=MAX_LENGTH):
    with torch.no_grad():
        decoder_output, loss, _, _ = seq2seq(input_batches, input_lengths, target_batches, target_lengths, batch_size, criterion)


        return decoder_output, loss.data

def main():

    lang, pairs = prepare_data("data/data.txt")
    train_lang, train_pairs = prepare_data("data/train.txt")
    test_lang, test_pairs = prepare_data("data/test.txt")
    MIN_COUNT = 5


    lang.trim(MIN_COUNT)
    train_lang.trim(MIN_COUNT)
    test_lang.trim(MIN_COUNT)

    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in lang.word2index:
                keep_input = False
                break

        for word in output_sentence.split(' '):
            if word not in lang.word2index:
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)

    for pair in train_pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in train_lang.word2index:
                keep_input = False
                break

        for word in output_sentence.split(' '):
            if word not in train_lang.word2index:
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from %d pairs to %d, %.4f of total" % (len(train_pairs), len(keep_pairs), len(keep_pairs) / len(train_pairs)))
    train_pairs = keep_pairs

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
    encoder = EncoderRNN(lang.n_words, hidden_size, n_layers, dropout=dropout)
    decoder = LuongAttnDecoderRNN(attn_model, hidden_size, lang.n_words, n_layers, dropout=dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    seq2seq = Seq2Seq(encoder, decoder, encoder_optimizer, decoder_optimizer)

    if USE_CUDA:
        seq2seq.cuda()


    # Keep track of time elapsed and running averages
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Begin!
    ecs = []
    dcs = []
    eca = 0
    dca = 0

    epoch = 0

    # early_stopping object의 초기화
    early_stopping = EarlyStopping(patience = patience, verbose = True)

    for epoch in range(1, n_epochs + 1):
        # Get training data for this cycle
        input_batches, input_lengths, _, target_batches, target_lengths = random_batch(batch_size, train_pairs, lang)

        test_input_batches, test_input_lengths, input_origin, test_target_batches, test_target_lengths = random_batch(batch_size, test_pairs, lang)

        # Run the train function
        _, loss, ec, dc = seq2seq(input_batches, input_lengths, target_batches, target_lengths, batch_size, criterion)

        #loss.backward()

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc

        decoder_output, val_loss = evaluate(
            test_input_batches, test_input_lengths, test_target_batches, test_target_lengths, seq2seq, criterion)

        if epoch % 100 == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)
            print("evaluation - epoch", epoch, " loss: ", val_loss)


            original = test_input_batches.transpose(1, 0)
            decoder_output = decoder_output.transpose(1, 0)
            #predict(original, decoder_output, lang)


            # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
            # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
            early_stopping(val_loss, seq2seq)

            if early_stopping.early_stop:
                print("Early stopping")
                break



main()