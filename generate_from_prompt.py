from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed

import yaml


def generation_example(model_name='TrainedModel',
                       seed=42,
                       input_string='this is a test',
                       bad_words=[]
):
    set_seed(seed)

    ###########################
    #LOAD MODEL AND TOKENIZER#
    ###########################
    # GPT2LMHeadModel is GPT2 but with an added language modeling 'head'
    # the head is just a linear layer on top which gives probabilities for each word in vocab
    # GPT2LMHeadModel also gives us the generate() method which picks sequences of words using a decoding method
    pt_model = GPT2LMHeadModel.from_pretrained(model_name)
    # The tokenizer turns input strings into tensors and output tensors into strings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #####################
    #ENCODE INPUT TOKENS#
    #####################
    # using tokenizer(raw_string) gives a batch output, but generate() can't handle batches without some extra steps
    # this is why we use tokenizer.encode()
    sample_input = tokenizer.encode(input_string, add_special_tokens=True, return_tensors='pt')
    print('Raw string: {}\n'.format(input_string))
    print('Encoded string: {}\n'.format(sample_input))

    ############################
    #GENERATE OUTPUT FROM INPUT#
    ############################
    if bad_words:
        bad_words = tokenizer(bad_words, add_special_tokens=True, return_tensors='pt')

    # this generates a sequence of output tokens using some decoding method.
    # see here for more info: https://huggingface.co/blog/how-to-generate
    # it's using top-k sampling with temperature
    pt_text = pt_model.generate(
        sample_input,
        max_length=256,
        temperature=0.3,
        repetition_penalty=1.15,
        top_k=60,
        do_sample=True,
        num_return_sequences=5,
        bad_words_ids=bad_words
    )
    print('Raw output from generation: {}\n'.format(pt_text))

    ######################
    #DECODE OUTPUT TOKENS#
    ######################
    pt_text = pt_text.squeeze_()
    for idx, generated_sequence in enumerate(pt_text):
        print("Generated message {}".format(idx))
        pt_text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        print('Output after tokenizer decoding:')
        print(pt_text)
        print('')


if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml'))['generation']
    generation_example()
