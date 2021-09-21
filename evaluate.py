from __future__ import unicode_literals, print_function, division
import random
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
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
"""

def main():
    lang, pairs, grouped = prepareData("data/test.txt")

    hidden_size = 256
    learning_rate = 0.01


    # 경로 지정
    PATH = "state_dict_model.pt"

    model = torch.load(PATH)

    training_pairs = [tensorsFromPair(random.choice(grouped), lang)]

    start = time.time()
    print_loss_total = 0  # print_every 마다 초기화


    with torch.no_grad():
        input_tensor = training_pairs[0][0]
        target_tensor = training_pairs[0][1]

        loss = model(input_tensor, target_tensor)
        print_loss_total += loss

        print('loss: ' , print_loss_total)

main()