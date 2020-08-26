# Overview

Have your friends ever been offline when you really wanted to chat? Have you ever felt irresponsible for not having a backup of your friends? Do you believe that piracy isn't theft?

Well pirate a copy of your friend's personality with FriendSimulator!

This project adapts example code from the Huggingface/transformers repo to fine-tune GPT2 on DMs from your friends. Then, you can either generate whole conversations from a prompt or simulate actual conversations. Do you ever worry you're annoying your friends? Well play as your friend and decide for yourself!

The procedure for using this software is loosely as follows:

1) Prepare the training data using DiscordChatExporter and dm_parser.py
2) Train the model using train.py
3) Simulate conversations using either generate_from_prompt.py or simulate_conversation.py

We'll talk about the steps in detail in the following sections.

Although the procedure described would use Discord DMs, any properly-formatted chat logs can be used as training data.

DISCLAIMER: Automated user accounts are against the Discord terms of service. Thus, using this code base with the linked Discord scraper is a hypothetical (and satirical) project. Additionally, using conversation data to train a neural net is morally grey. Talk with your friends about this hypothetical project before deciding whether you would hypothetically use this code.

## General Preparation

I recommend you create a Python 3.x environment for this procedure. You will want to "pip install torch" to get the PyTorch package.

You will also want to install the Huggingface/transformers package from the source (i.e. using the latest github commit). This can be done with the commands:

``git clone https://github.com/huggingface/transformers.git

cd transformers

pip install -e .``

After this, you should be just about ready!

## Preparing Training Data

[Discord Chat Exporter][1] by Tyrrrz is a great tool for hypothetically scraping Discord logs into a convenient format.

Here is the currently implemented procedure:

1) Use Discord Chat Exporter to scrape DMs. Make sure to scrape them into a CSV file.
2) Open *dm_parser.py* and set the *input_filename* variable to the CSV's filename. Take note of the "output_filename"
3) Run *dm_parser.py* and verify the outputs have been produced. The outputs are three txt files. GPT2 will train and test on two of these. The third file holds usernames and is also needed for the training program. The train/test ratio for the logs is 95/5 because I'm a bad scientist.

Remember that scraping Discord DMs is against the Discord terms of service, so this is a hypothetical procedure.

## Training a Model

To train a new GPT model, you will first want to open the *training_config.json* file. Here are the most important parameters:

* *model_type* is the desired GPT2 model. The sizes are *gpt2*, *gpt2-medium*, *gpt2-large*, *gpt2-xl*. Some of these may be too big for your computer to train.
* *model_name_or_path* should be the same as *model_type*
* *usernames_file* is generated from the parser and contains a list of usernames in the chat
* *output_dir* is the folder where the trained neural net will be stored. You will reference this folder later when using the net to simulate conversation.
* *train_data_file* and *test_data_file* should be the files produced by *dm_parser.py*

Here are some less important but still interesting parameters:

* *overwrite_output_dir* determines whether to overwrite any existing NN in the *output_dir*
* *block_size* is the length of text samples the NN will train on.
* *gradient_accumulation_steps* can be increased to make the memory usage a little more efficient
* *no_cuda* forces the training procedure to not use CUDA (i.e. not to use your GPUs). I have to use this because my GPU is not big enough for training most GPT2 networks.

Once you've verified your parameters are acceptable, simply run the *train.py* file. When it finishes, you will have a trained NN in the designated *output_dir*, and you're ready to start simulating!

## Simulating With a Model

The two simulation files are *generate_from_prompt.py* and *simulate_conversation.py*.

Before you run these, ensure the *generation_config.json* is properly configured. Particularly, *model_name_or_path* must be the folder of the desired NN. For example, if the folder is "Steven" and it's located in the same location as *generate_from_prompt.py* and *simulate_conversation.py*, you would enter *Steven/*

you do not need to change the *model_type* parameter, regardless of which GPT2 size you use.

I recommend researching and experimenting with *temperature*, *repetition_penalty* and *k*. These parameters can affect the quality of the outputs. The defaults are what I found to be effective.

NOTE! It's better to turn up *num_return_sequences* and turn down *length* when running *simulate_conversation.py*. This helps with speed and gives more possible AI responses.

# Common Issues

### The training terminates with an error, "CUDA out of memory"

Your GPU can't handle your neural net. You can try using a smaller GPT2, lowering batch sizes, increasing gradient accumulation, or activating no_cuda in the config and training on CPU

### The training suddenly terminates without any information

I'm pretty sure you ran out of RAM. Try everything from the previous question. Barring that, get more RAM. You can also train this using Google's Colab. I was able to train gpt2-large using a Pro account.

### The output is not convincing

Try tweaking the parameters in *generation_config.json*. Also note that I only got convincing results when using gpt2-large.

### The code is poorly written

Yeah

[1]:https://github.com/Tyrrrz/DiscordChatExporter