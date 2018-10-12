from .unet import vanilla_unet


def make_model(network, input_shape, with_sigmoid=True, random_state=None):
    if network == 'vanilla_unet':
        return vanilla_unet(input_shape, with_sigmoid, random_state)