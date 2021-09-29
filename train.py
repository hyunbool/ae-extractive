# ref: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

from torch import optim
from model import *

USE_CUDA=True
# set cuda device and seed
if USE_CUDA:
    torch.cuda.set_device(0)
torch.cuda.manual_seed(1)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

teacher_forcing_ratio = 0.5

attn_model = 'dot'
hidden_size = 500
n_layers = 2
dropout = 0.1
batch_size = 50


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 50000

print_every = 100
evaluate_every = 100



def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data, ec, dc


def evaluate(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, criterion, max_length=MAX_LENGTH):
    with torch.no_grad():
        loss = 0  # Added onto for each word

        # Run words through encoder
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

        max_target_length = max(target_lengths)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

        # Move new Variables to CUDA
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()

        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t]  # Next input is current target

        # Loss calculation and backpropagation
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )


        return loss.data, all_decoder_outputs


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


    train_lang.trim(MIN_COUNT)
    test_lang.trim(MIN_COUNT)

    keep_pairs = []

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
    encoder = EncoderRNN(train_lang.n_words, hidden_size, n_layers, dropout=dropout)
    decoder = LuongAttnDecoderRNN(attn_model, hidden_size, train_lang.n_words, n_layers, dropout=dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()



    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Begin!
    ecs = []
    dcs = []
    eca = 0
    dca = 0

    epoch = 0
    print("train_pairs: ", str(train_pairs[:10]))
    print("test_pairs: ", str(test_pairs[:10]))
    while epoch < n_epochs:
        epoch += 1

        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, train_pairs, train_lang)
        test_input_batches, test_input_lengths, test_target_batches, test_target_lengths = random_batch(batch_size, test_pairs, test_lang)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc


        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % evaluate_every == 0:
            # Run the train function
            loss, decoder_output = evaluate(
                test_input_batches, test_input_lengths, test_target_batches, test_target_lengths,
                encoder, decoder, criterion)
            print("evaluation - epoch", epoch, " loss: ", loss.item())

            original = test_input_batches.transpose(1, 0)
            decoder_output = decoder_output.transpose(1, 0)
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
            for t in original:
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


main()