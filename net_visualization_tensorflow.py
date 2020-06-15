import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    X = tf.convert_to_tensor(X)
    N, H, W, C = X.shape
    with tf.GradientTape() as t:
        t.watch(X)
        s = model(X)
        index = tf.stack((tf.range(N), y), axis=1)
        correct_s = tf.gather_nd(s, index)
        dx = t.gradient(correct_s, X)
        
    saliency = tf.math.reduce_max(tf.abs(dx), axis=3)
    
    dx_pos = tf.math.divide(tf.math.add(dx, tf.abs(dx)), 2)
    saliency_pos = tf.math.reduce_max(dx_pos, axis=3)
    
    dx_neg = tf.math.divide(tf.math.subtract(dx, tf.abs(dx)), 2)
    saliency_neg = tf.math.reduce_max(dx_neg, axis=3)
   #################################
    return saliency

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, a numpy array of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """

    # Make a copy of the input that we will modify
    X_fooling = X.copy()

    # Step size for the update
    learning_rate = 1

    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient *ascent* on the target class score, using #
    # the model.scores Tensor to get the class scores for the model.image.   #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop, where in each iteration, you make an     #
    # update to the input image X_fooling (don't modify X). The loop should      #
    # stop when the predicted class for the input is the same as target_y.       #
    #                                                                            #
    # HINT: Use tf.GradientTape() to keep track of your gradients and            #
    # use tape.gradient to get the actual gradient with respect to X_fooling.    #
    #                                                                            #
    # HINT 2: For most examples, you should be able to generate a fooling image  #
    # in fewer than 100 iterations of gradient ascent. You can print your        #
    # progress over iterations to check your algorithm.                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_fooling = tf.convert_to_tensor(X_fooling)
    num_iter = 100
    for i in range(num_iter):
        predict_scores = model.predict(X_fooling)  # dtype: nparray
        predict_class = tf.math.argmax(predict_scores[0]).numpy()  # dtype: nparray
        if predict_class == target_y:
            break            
        with tf.GradientTape() as t:
            t.watch(X_fooling)
            s = model(X_fooling)
            correct_s = tf.gather_nd(s, tf.convert_to_tensor([[0,target_y]]))
            dx_fooling = t.gradient(correct_s, X_fooling)
            
        dx_fooling_norm = tf.math.divide(dx_fooling, tf.norm(dx_fooling))
        X_fooling += tf.math.multiply(dx_fooling_norm, learning_rate)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling

def class_visualization_update_step(X, model, target_y, l2_reg, learning_rate):
    ########################################################################
    # TODO: Compute the value of the gradient of the score for             #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. You should use   #
    # the tf.GradientTape() and tape.gradient to compute gradients.        #
    #                                                                      #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    with tf.GradientTape() as tape:
            tape.watch(X)
            s = model(X)
            correct_s = tf.gather_nd(s, tf.convert_to_tensor([[0,target_y]]))
            reg = tf.math.multiply(tf.math.pow(tf.norm(X) ,2), l2_reg)
            correct_s_reg = tf.math.subtract(correct_s, reg)
            dx = tape.gradient(correct_s_reg, X)
            
    X += tf.math.multiply(dx, learning_rate)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return X

def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: Tensor of shape (N, H, W, C)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new Tensor of shape (N, H, W, C)
    """
    if ox != 0:
        left = X[:, :, :-ox]
        right = X[:, :, -ox:]
        X = tf.concat([right, left], axis=2)
    if oy != 0:
        top = X[:, :-oy]
        bottom = X[:, -oy:]
        X = tf.concat([bottom, top], axis=1)
    return X
