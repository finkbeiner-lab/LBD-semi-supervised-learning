from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attr = ig.attribute(inputs=(images, cat_vars), target=label)