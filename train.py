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


        return decoder_output, loss.data


def evaluate_randomly(encoder, decoder, pairs, lang):
    [input_sentence, target_sentence] = random.choice(pairs)
    output_words, _ = evaluate(encoder, decoder, input_sentence, lang)

    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)



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
        input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, train_pairs, lang)
        test_input_batches, test_input_lengths, test_target_batches, test_target_lengths = random_batch(batch_size, test_pairs, lang)

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

            print(decoder_output)
            results = []
            for t in decoder_output:
                for di in range(MAX_LENGTH):
                    decoded_words = []
                    # Choose top word from output
                    topv, topi = t.data.topk(1)
                    ni = topi[0][0].item()
                    print(ni)
                    if ni == EOS_token:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(lang.index2word[ni])
                results.append(decoded_words)
            print(results)

            # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
            # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
            early_stopping(val_loss, seq2seq)

            if early_stopping.early_stop:
                print("Early stopping")
                break



main()