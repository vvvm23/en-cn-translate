import torch
import sys

from dataloader import LangDataset, remove_punctuation

# Constant parameters
TRY_CUDA = True
MAX_LEN = 44

# Get CUDA device if available
device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'})")

# Initialise dataset in order to use tokenizers
dataset = LangDataset("data/cmn.txt")
print(f"> Initialised tokenizer.")

# Load the pretrained model and switcht to evaluation mode
model = torch.load(sys.argv[1]).to(device)
model.eval()
print(f"> Loaded model {sys.argv[1]} from file.")

while True:
    # Get an english sentence from the user
    print(f"\n> Enter English sentence:")
    eng_input = input("> ")
    
    # If the input is empty this indicates we wish to exit.
    if eng_input == '':
        print("> Empty input. Exiting..")
        break

    eng_input = remove_punctuation(eng_input).lower()
    try:
        # Attempt to tokenize the input sentence
        eng_input = torch.tensor([1] + dataset.token_in._sentence_to_index(eng_input) + [2] + [0]*(dataset.max_in - len(eng_input))).to(device)
    except KeyError as e:
        # If we don't recognise a token, continue the loop to prompt for a new input
        print("! One or more words was not recognised!")
        continue
    except e:
        # If there is an actual error we should instead raise it
        raise e

    # Initialise tgt to just the SOS token
    tgt = torch.tensor([1] + [0]*(dataset.max_out+2)).to(device)

    # Add a batch dimension to both src and tgt
    eng_input, tgt = eng_input.unsqueeze(0), tgt.unsqueeze(0)

    # Get the next prediction until we get to max length or the model predicts EOS
    for i in range(MAX_LEN):
        out = model(eng_input, tgt)
        argmax_out = torch.argmax(out[0, i, :], dim=-1)
        tgt[0, i+1] = argmax_out

        if argmax_out == 2:
            break

    # Convert chinese output from tokens to characters and change to string
    cn_output = ''.join([dataset.token_out.index_word[w] if w not in [0, 1, 2] else '' for w in tgt.squeeze().tolist()])
    print(f"> {cn_output}")
