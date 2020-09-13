import torch
import sys

from dataloader import LangDataset, remove_punctuation

TRY_CUDA = True
MAX_LEN = 44

device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'})")

dataset = LangDataset("data/cmn.txt")
print(f"> Initialised tokenizer.")

model = torch.load(sys.argv[1]).to(device)
print(f"> Loaded model {sys.argv[1]} from file.")

while True:
    print(f"\n> Enter English sentence:")
    eng_input = input("> ")
    
    if eng_input == '':
        print("> Empty input. Exiting..")
        break

    eng_input = remove_punctuation(eng_input).lower()
    eng_input = eng_input.split(' ')
    try:
        eng_input = torch.tensor([1] + [dataset.token_in.word_index[w] for w in eng_input] + [2] + [0]*(dataset.max_in - len(eng_input))).to(device)
    except KeyError as e:
        print("! One or more words was not recognised!")
        continue
    except e:
        raise e

    tgt = torch.tensor([1] + [0]*(dataset.max_out+2)).to(device)

    eng_input, tgt = eng_input.unsqueeze(0), tgt.unsqueeze(0)

    for i in range(MAX_LEN):
        out = model(eng_input, tgt)
        argmax_out = torch.argmax(out[0, i, :], dim=-1)
        tgt[0, i+1] = argmax_out

        if argmax_out == 2:
            break

    cn_output = ''.join([dataset.token_out.index_word[w] if w not in [0, 1, 2] else '' for w in tgt.squeeze().tolist()])
    print(f"> {cn_output}")
