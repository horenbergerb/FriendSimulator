from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed
import yaml


def generate_from_prompt(model_name='TrainedModel',
                         seed=42,
                         input_string='this is a test',
                         temperature=0.4,
                         max_length=256,
                         top_k=50,
                         top_p=.95,
                         num_return_sequences=2,
                         repetition_penalty=1.15,
                         bad_words=[]):

    set_seed(seed)

    ############################
    # LOAD MODEL AND TOKENIZER #
    ############################
    # The tokenizer turns input strings into tensors and output tensors into strings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT2LMHeadModel is GPT2 but with an added language modeling 'head'
    # the head is just a linear layer on top which gives probabilities for each word in vocab
    # GPT2LMHeadModel also gives us the generate() method which picks sequences of words using a decoding method
    # i specified the pad_token_id to stop an annoying warning from getting printed
    pt_model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

    #######################
    # ENCODE INPUT TOKENS #
    #######################
    # using tokenizer(raw_string) gives a batch output, but generate() can't handle batches without some extra steps
    # this is why we use tokenizer.encode()
    sample_input = tokenizer.encode(input_string, add_special_tokens=True, return_tensors='pt')

    ##############################
    # GENERATE OUTPUT FROM INPUT #
    ##############################
    if bad_words:
        bad_words = tokenizer(bad_words, add_special_tokens=True, return_tensors='pt')['input_ids'].tolist()

    # this generates a sequence of output tokens using some decoding method.
    # see here for more info: https://huggingface.co/blog/how-to-generate
    # it's using top-k sampling with temperature
    pt_text = pt_model.generate(
        sample_input,
        max_length=max_length,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        bad_words_ids=bad_words
    )

    ########################
    # DECODE OUTPUT TOKENS #
    ########################
    pt_text = pt_text.squeeze_()
    for idx, generated_sequence in enumerate(pt_text):
        pt_text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        print(pt_text)
        print('')


if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml'))['generation']
    generate_from_prompt(**config)
