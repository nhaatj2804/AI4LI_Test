import pandas as pd
import torch


# from encoder_architecture import Encoder
# from decoder_architecture import Decoder
# ENCODER = Encoder()
# ENCODER.load_state_dict(torch.load('encoder_final.pth', map_location=torch.device('cpu'), weights_only=False))

# DECODER = Decoder()
# DECODER.load_state_dict(torch.load('decoder_final.pth', map_location=torch.device('cpu'), weights_only=False))

DATA = pd.read_csv('seq2seq/train_data.tsv', sep='\t', header=None, names=['input', 'target'])
DATA = DATA[1:]

def tokenize(text):
    return text.lower().strip().split()

all_input_words = []
all_target_words = []
for idx,row in DATA.iterrows():
    all_input_words.extend(tokenize(row['input']))
    all_target_words.extend(tokenize(row['target'])) # Đảm bảo đây là row['target']
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

input_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(set(all_input_words))

target_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(set(all_target_words))

input_word2idx = {'<PAD>': 0,
 '<SOS>': 1,
 '<EOS>': 2,
 'nice': 3,
 'meet': 4,
 'please': 5,
 'you': 6,
 'name': 7,
 't': 8,
 'm': 9,
 'hello': 10,
 'o': 11,
 'sit': 12,
 'yes': 13}

input_idx2word = {'<PAD>': 0,
 '<SOS>': 1,
 '<EOS>': 2,
 'nice': 3,
 'meet': 4,
 'please': 5,
 'you': 6,
 'name': 7,
 't': 8,
 'm': 9,
 'hello': 10,
 'o': 11,
 'sit': 12,
 'yes': 13} # Cách này đúng

target_word2idx ={'<PAD>': 0,
 '<SOS>': 1,
 '<EOS>': 2,
 'meet': 3,
 'is': 4,
 'nice.': 5,
 'please': 6,
 "that's": 7,
 'down,': 8,
 'tom!': 9,
 'are': 10,
 'name': 11,
 'yes,': 12,
 "let's": 13,
 'you,': 14,
 'meet!': 15,
 'nice': 16,
 'you!': 17,
 'how': 18,
 'sit': 19,
 'yes.': 20,
 'down.': 21,
 'please.': 22,
 'your': 23,
 'name?': 24,
 "what's": 25,
 'to': 26,
 'you?': 27,
 'my': 28,
 'please!': 29,
 'tom.': 30,
 'you.': 31,
 'hello!': 32,
 'hello,': 33}
target_idx2word = {0: '<PAD>',
 1: '<SOS>',
 2: '<EOS>',
 3: 'meet',
 4: 'is',
 5: 'nice.',
 6: 'please',
 7: "that's",
 8: 'down,',
 9: 'tom!',
 10: 'are',
 11: 'name',
 12: 'yes,',
 13: "let's",
 14: 'you,',
 15: 'meet!',
 16: 'nice',
 17: 'you!',
 18: 'how',
 19: 'sit',
 20: 'yes.',
 21: 'down.',
 22: 'please.',
 23: 'your',
 24: 'name?',
 25: "what's",
 26: 'to',
 27: 'you?',
 28: 'my',
 29: 'please!',
 30: 'tom.',
 31: 'you.',
 32: 'hello!',
 33: 'hello,'} # Cách xây dựng trực tiếp target_idx2word từ target_vocab


def encode_input_for_eval(sentence):
    return [input_word2idx.get(word, input_word2idx['<PAD>']) for word in sentence.lower().split()]



def evaluate(encoder, decoder, sentence, input_word2idx = input_word2idx, target_word2idx = target_word2idx, target_idx2word = target_idx2word, encode_input_func = encode_input_for_eval, max_len=20, device='cpu'): # đổi tên encode_input để tránh nhầm lẫn
    encoder.eval()
    decoder.eval()

    # Sử dụng hàm encode_input_func được truyền vào
    tokens = encode_input_func(sentence) # Đổi tên ở đây nếu cần
    print(f"Input sentence: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"Input vocab size for check: {len(input_word2idx)}")


    inputs = torch.tensor(tokens).unsqueeze(0).to(device)
    lengths = torch.tensor([len(tokens)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = encoder(inputs, lengths)
        decoder_input = torch.tensor([target_word2idx['<SOS>']]).to(device)
        decoder_hidden = hidden

        decoded_words = []
 

        for i in range(max_len):
            output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            top1 = output.argmax(1).item()
   

            if top1 not in target_idx2word:
                print(f"----> CRITICAL: Index {top1} is NOT a key in target_idx2word (max index is {len(target_idx2word)-1})")
            
            if top1 == target_word2idx['<EOS>']:
                print("Predicted EOS token.")
                break

            word = target_idx2word.get(top1, '<UNK>') # Default to <UNK>
            if word == '<UNK>':
                print(f"----> WARNING: Predicted token is <UNK> for index {top1}")
            decoded_words.append(word)

            decoder_input = torch.tensor([top1]).to(device)
    
    return ' '.join(decoded_words)

# if __name__ == "__main__":
#     print(evaluate(ENCODER, DECODER, "meet T O M"))
