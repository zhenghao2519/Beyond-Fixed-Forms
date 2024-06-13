################################################################################################################
# Utility functions, can move helper functions here to make original code less cluttered

# useful if predict_extended took a list of base_prompt

from typing import List

def print_dict(dict_descriptors):
    '''
    print the dict of descriptors, where
    dict_descriptors = {class:[extended captions for the class]}
    '''
    for k, v in dict_descriptors.items():
        for descriptor in v:
            print(f" {k} - {descriptor}")

def combine_base_and_descriptors(dict_descriptors) ->List[List[str]]:
    '''
    args: dict_descriptors = {class:[extended captions for the class]}
    returns: combined = [[class, descriptors], [class, descriptors], ...]
    '''
    combined = []
    for k, v in dict_descriptors.items():
        extended = []
        for descriptor in v:
            extended.append(f"{k}, {descriptor}")
        combined.append(extended)
    return combined