from __future__ import unicode_literals, print_function, division
from torch import optim
import torch
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "entire_model.pt"


def evaluate(encoder, decoder, sentence, lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, grouped, lang, n=10):
    for i in range(n):
        pair = random.choice(grouped)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def main():
"""
    lang, pairs, grouped = prepareData("data/test.txt")

    hidden_size = 256
    learning_rate = 0.01

    encoder1 = EncoderRNN(lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, lang.n_words, dropout_p=0.1).to(device)
    encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=learning_rate)

    checkpoint = torch.load(PATH)


    #encoder1.load_state_dict(checkpoint['encoder_state_dict'])
   # attn_decoder1.load_state_dict(checkpoint['decoder_state_dict'])
    #encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    #decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])

    #evaluateRandomly(encoder1, attn_decoder1, grouped, lang)
"""
    # 경로 지정
    PATH = "state_dict_model.pt"

    model = Net()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    main()