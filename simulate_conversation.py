from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed
import yaml
import torch

import csv


def simulate_conversation(model_name='TrainedModel',
                          usernames_dir='DMsParsedUsernames.txt',
                          seed=42,
                          temperature=0.4,
                          max_length=128,
                          top_k=50,
                          top_p=.95,
                          num_return_sequences=2,
                          repetition_penalty=1.15,
                          bad_words=[]):

    if bad_words is None:
        bad_words = []

    set_seed(seed)

    print('Loading model...')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pt_model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

    print('Loading usernames...')

    usernames = []
    with open(usernames_dir) as usernames_file:
        usernames_reader = csv.reader(usernames_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        for row in usernames_reader:
            usernames.append('<'+row[0]+'>')

    print("Select a character:")
    player = ""
    for name in usernames:
        print(name)
    while player not in usernames:
        player = input("Type your selection:")
    bad_words.append(player)

    # update bad words so gpt2 won't speak as the player
    if bad_words:
        bad_words = tokenizer(bad_words, add_special_tokens=True, return_tensors='pt')['input_ids'].tolist()

    print('')
    print('Type "/exit" to quit.')
    print('Press enter without typing to let GPT2 continue.')
    print('Have fun!')
    print('')

    gpt2_input = ''
    while True:

        user_input = input(player + " ")
        if user_input == '/exit':
            break

        if (user_input != ""):
            gpt2_input += (player + " " + user_input)
            gpt2_input += "\n"

        encoded_gpt2_input = tokenizer.encode(gpt2_input, add_special_tokens=True, return_tensors='pt')
        print(encoded_gpt2_input)
        if len(encoded_gpt2_input[0]) > max_length//2:
            encoded_gpt2_input = encoded_gpt2_input[..., len(encoded_gpt2_input)-(max_length//2):]
            print(encoded_gpt2_input)
            gpt2_input = tokenizer.decode(encoded_gpt2_input[0], clean_up_tokenization_spaces=True)

        pt_text = pt_model.generate(
            encoded_gpt2_input,
            max_length=max_length,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            bad_words_ids=bad_words
        )

        pt_text = pt_text.squeeze_()
        pt_text = tokenizer.decode(pt_text, clean_up_tokenization_spaces=True)
        new_content = pt_text[len(gpt2_input):].split('\n')[0]
        print(new_content)
        gpt2_input += new_content + '\n'


if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml'))['conversation']
    simulate_conversation(**config)
