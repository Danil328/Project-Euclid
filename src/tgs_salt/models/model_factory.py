from .unet import vanilla_unet


def make_model(network, input_shape, random_state=None):
    if network == 'vanilla_unet':
        return vanilla_unet(input_shape, random_state)
    else:
        raise ValueError('Unknown network {0}'.format(network))