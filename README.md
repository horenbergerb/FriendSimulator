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

DISCLAIMER: Automated user accounts are against the Discord terms of service. Thus, using this code base with the linked Discord scraper is a hypothetical (and satirical) project. Additionally, using conversation data to train a neural net is morally grey. Don't go around pirating personalities. That's dystopian cyberpunk stuff.

## General Preparation

I recommend you create a Python 3.x environment for this procedure. You can install the requirements using `pip install -r requirements.txt`

You may need to install the Huggingface/transformers package from the source (i.e. using the latest github commit). This can be done with the commands:

```git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

After this, you should be just about ready!

## Preparing Training Data

[Discord Chat Exporter][1] by Tyrrrz is a great tool for hypothetically scraping Discord logs into a convenient format.

Here is the currently implemented procedure:

1) Use Discord Chat Exporter to scrape DMs. Make sure to scrape them into a CSV file.
2) Run `dm_parser.py filename` where 'filename' is the CSV's filename.
3) The outputs are three txt files. Update the training section of the config.yaml file to point to these. The train/test ratio used by dm_parser is 95/5 because I'm a bad scientist.

## Training a Model

To fine-tune a GPT2 model, you will first want to open the *config.yaml* file. Here are the most important parameters:

* *train_dir*, *eval_dir*, *usernames_dir* will all need to be updated to match the outputs from dm_parser.py
* *model_name* is the desired GPT2 model. The sizes are *gpt2*, *gpt2-medium*, *gpt2-large*, *gpt2-xl*. Some of these may be too big for your computer to train.

Here are some less important but still interesting parameters:

* *overwrite_output_dir* determines whether to overwrite any existing NN in the * *output_dir* is where the fine-tuned network will be stored
* *block_size* is the length of text samples the network will train on.
* *gradient_accumulation_steps* can be increased to make the memory usage a little more efficient
* *no_cuda* forces the training procedure to not use CUDA (i.e. not to use your GPUs). I have to use this because my GPU is not big enough for training most GPT2 networks.

Once you've verified your parameters are acceptable, simply run the *train.py* file. When it finishes, you will have a trained NN in the designated *output_dir*, and you're ready to start simulating!

## Simulating With a Model

The two simulation files are *generate_from_prompt.py* and *simulate_conversation.py*.

Before you run these, ensure the generation section of *config.yaml* is properly configured. Particularly, *model_name* must be the folder of the desired NN. For example, if the folder is "TrainedModel" and it's located in the same location as *generate_from_prompt.py* and *simulate_conversation.py*, you would enter *TrainedModel*

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