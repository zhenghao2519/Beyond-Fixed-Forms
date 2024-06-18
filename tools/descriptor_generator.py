################################################################################################################
# This file contains the functions that generate descriptors for the prompts.
import openai
import os
from dotenv import load_dotenv
from typing import List

#set OpenAI API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')  # stored in .env file on the machine for security

# Utility functions
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

### Descriptor Makers.
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
    return descriptors_structured

def generate_descriptors_waffle(base_prompt) ->List[str]:
    # TBD
    descriptors = []
    return descriptors

def generate_descriptors_gpt(base_prompt) ->List[str]:
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
    return descriptors

def generate_descriptors_waffle_and_gpt(base_prompt) ->List[str]:
    # TBD
    descriptors = []
    return descriptors

# And so on, TODO: fill in the functions

# Main function for testing
def main():
    base_prompt = input("Enter the base prompt(categorie name): ")
    descriptors = generate_descriptors_gpt(base_prompt)
    print(descriptors)

if __name__ == '__main__':
    main()