################################################################################################################
# This file contains the functions that generate descriptors for the prompts.
import openai
import os
import numpy as np
import string
from dotenv import load_dotenv
from typing import List
import yaml
from munch import Munch

# List of methods available to use.
METHODS = [
    'toy',
    'gpt',
    'waffle',
    'waffle_and_gpt'
]

#%% =========================Utility functions======================================
def wordify(string: str):
    word = string.replace('_', ' ')
    return word

def modify_descriptor(descriptor: str, apply_changes: bool):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor

def make_descriptor_sentence(descriptor: str):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
    
def string_to_list(description):
        return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]

#%% =======================Descriptor generators======================================
def structured_descriptor_builder(descriptor, cls):
    # TODO: move it to config
    pre_descriptor_text = ''
    label_before_text = 'A photo of a '
    label_after_text = '.'
    descriptor_separator = ', '
    apply_descriptor_modification = True

    return f"{pre_descriptor_text}{label_before_text}{wordify(cls)}{descriptor_separator}{modify_descriptor(descriptor, apply_descriptor_modification)}{label_after_text}"
    # generic_descriptor_builder = lambda item, cls: f"{opt.pre_descriptor_text}{opt.label_before_text}{wordify(cls)}{opt.descriptor_separator}{item}{opt.label_after_text}"    

def toy_descriptors(base_prompt):          
    '''
    toy function to mock the extending of prompts
    '''

    descriptors = ["aks@, pg2f","foot loud", "w6y#, d4e^", "r1q$, m3b@", "r1q$, m3b@", "q4g/, h9m~", "s2t=, i1p-", " g8c, a3v+", " o9n_, f0h?", "k2x%, u5j&", "m3b@, l7z!"]

    descriptors_structured = [structured_descriptor_builder(descriptor, base_prompt) for descriptor in descriptors]
    descriptions = {base_prompt: descriptors_structured}
    return descriptions

def generate_descriptors_waffle(base_prompt) -> str:
    def random_char(num_word, word_length):
        # Generate a random char seq of a given length
        res = []
        for _ in range(num_word):
            res.append(''.join(np.random.choice(list(string.ascii_letters + string.digits + string.punctuation), word_length)))
        res_str = ' '.join(res)
        return res_str
        # return ''.join(np.random.choice(list(string.digits + string.ascii_letters + string.punctuation), length))

    def random_word(num_word, word_list):
        
        
        res = []
        for _ in range(num_word):
            res.append(''.join(np.random.choice(word_list,1)))
        res_str = ' '.join(res)
        return res_str

    # Sample the number of descriptors from a Poisson distribution
    num_descriptors_pairs = 15
    num_words = 2
    word_length = 5
      
    descriptors = []
    
    # Load the word list
    import pickle as pkl
    print("current working directory: ", os.getcwd())
    word_list = pkl.load(open('tools/waffle_word_list.pkl', 'rb'))
    # print(word_list)
    word_list = [x[:word_length] for x in word_list]
    
    for _ in range(num_descriptors_pairs):
        descriptors.append(random_word(num_words, word_list))
        descriptors.append(random_char(num_words, word_length))

    descriptors_structured = [structured_descriptor_builder(descriptor, base_prompt) for descriptor in descriptors]
    descriptions = {base_prompt: descriptors_structured}
    return descriptions

def generate_descriptors_gpt(base_prompt) ->List[str]:
    #set OpenAI API key
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')  # stored in .env file on the machine for security

    prompt_template = f"""
    Q: What are useful visual features for distinguishing a {base_prompt} in a photo?
    A: There are several useful visual features to tell there is a {base_prompt} in a photo:
    -
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": prompt_template}
        ],
        temperature=0.5, # randomness of the completions
        max_tokens=100  # maximum number of tokens to generate
    )
    
    descriptors = string_to_list(response.choices[0].text)
    descriptors_structured = [structured_descriptor_builder(descriptor, base_prompt) for descriptor in descriptors]
    descriptions = {base_prompt: descriptors_structured}
    return descriptions

def generate_descriptors_waffle_and_gpt(base_prompt) ->List[str]:
    gpt_descriptor = generate_descriptors_gpt(base_prompt)
    waffle_descriptor = generate_descriptors_waffle(base_prompt)
    descriptors = gpt_descriptor + waffle_descriptor
    descriptors_structured = [structured_descriptor_builder(descriptor, base_prompt) for descriptor in descriptors]
    descriptions = {base_prompt: descriptors_structured}
    return descriptions

# TODO: make generators more general, allowing List(str) as input

def descr_generator_selector(base_prompt, method):
    if method == 'toy':
        return toy_descriptors(base_prompt)
    elif method == 'gpt':
        return generate_descriptors_gpt(base_prompt)
    elif method == 'waffle':
        return generate_descriptors_waffle(base_prompt)
    elif method == 'waffle_and_gpt':
        return generate_descriptors_waffle_and_gpt(base_prompt)
    else:
        raise ValueError("Method not found. Please choose from the following methods: ", METHODS)

#%% ===================Main function for testing======================================
def main():
    cfg = Munch.fromDict(yaml.safe_load(open('/home/jie_zhenghao/Beyond-Fixed-Forms/configs/config.yaml', "r").read()))
    base_prompt = input("Using generator specified in config.yaml.\nPlease enter the base prompt (categorie name): ")
    descriptors = descr_generator_selector(base_prompt, cfg.descriptor_generator)
    print(descriptors)

if __name__ == '__main__':
    main()