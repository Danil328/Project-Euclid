def freeze_model(model, freeze_before_layer):

    if freeze_before_layer == 'all':
        for layer in model.layers:
            layer.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, layer in enumerate(model.layers):
            if layer.name == freeze_before_layer:
                freeze_before_layer_index = i
        for layer in model.layers[:freeze_before_layer_index + 1]:
            layer.trainable = False

def unfreeze_model(model):

    for layer in model.layers:
        layer.trainable = True