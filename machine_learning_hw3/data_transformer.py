from autograd import numpy as np
import edgebased_feature_extractor

# edge-based feature extraction
def edge_transformer(x):
    # edge-based directions for kernel-based feature extraction
    kernels = np.array([
           [[-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]],

           [[-1, -1,  0],
            [-1,  0,  1],
            [ 0,  1,  1]],

            [[-1,  0,  1],
            [-1,  0,  1],
            [-1,  0,  1]],

           [[ 0,  1,  1],
            [-1,  0,  1],
            [-1, -1,  0]],

           [[ 1,  0, -1],
            [ 1,  0, -1],
            [ 1,  0, -1]],

           [[ 0, -1, -1],
            [ 1,  0, -1],
            [ 1,  1,  0]],

           [[ 1,  1,  1],
            [ 0,  0,  0],
            [-1, -1, -1]],

           [[ 1,  1,  0],
            [ 1,  0, -1],
            [ 0, -1, -1]]])    

    # compute edge-based features
    demo = edgebased_feature_extractor.tensor_conv_layer()
    x_transformed = demo.conv_layer(x.T,kernels).T
    return x_transformed