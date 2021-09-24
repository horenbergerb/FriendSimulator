from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

import yaml
import math
import os
import csv
import torch


def fine_tune_gpt2(model_name='gpt2-large',
                   train_dir='DMsParsedTrain.txt',
                   eval_dir='DMsParsedTest.txt',
                   usernames_dir='DMsParsedUsernames.txt',
                   block_size=256,
                   plm_probability=1/6,
                   output_dir='TrainedModel',
                   overwrite_output_dir=False,
                   per_device_train_batch_size=1,
                   per_device_eval_batch_size=1,
                   gradient_accumulation_steps=16,
                   no_cuda=False,
                   num_train_epochs=3.0):

    torch.cuda.empty_cache()
    ############################
    # LOAD MODEL AND TOKENIZER #
    ############################
    pt_model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=block_size, padding=True, truncate=True)
    special_tokens = []
    with open(usernames_dir) as usernames_file:
        usernames_reader = csv.reader(usernames_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        for row in usernames_reader:
            special_tokens.append('<'+row[0]+'>')
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(len(tokenizer))
    pt_model.resize_token_embeddings(len(tokenizer))

    block_size = min(block_size, tokenizer.model_max_length)

    ###############################
    # LOAD TRAINING AND EVAL DATA #
    ###############################
    train_data = TextDataset(tokenizer=tokenizer, file_path=train_dir, block_size=block_size, overwrite_cache=False)
    eval_data = TextDataset(tokenizer=tokenizer, file_path=eval_dir, block_size=block_size, overwrite_cache=False)
    # a collator turns a list of dataset elements into a batch
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        do_eval=True,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        no_cuda=no_cuda,
        num_train_epochs=num_train_epochs,
        overwrite_output_dir=overwrite_output_dir
    )

    #############################
    # PREPARE TRAINER AND TRAIN #
    #############################
    trainer = Trainer(
        model=pt_model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    results = {}
    if training_args.do_eval:
        print("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        with open(output_eval_file, "w") as writer:
            print("***** Eval results *****")
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml'))['training']
    fine_tune_gpt2(**config)
