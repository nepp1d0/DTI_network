import torch


def extract_embedding(encoded_sample,  model):
    '''
    :param encoded_sample: tensor to feed model, output of tokenizer
    :param model: transformer model from Hugging Face
    :return: output from pooler.output of an instance of Bert model from Hugging Face
    '''
    # Compute model representation
    with torch.no_grad():
        output = model(**encoded_sample)

    '''layers= [-2, -1]
    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()'''

    return output.pooler_output