---------------------------------------------------------------------------------------------------

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


This code requires explanation. We use the 'sparse_categorical_crossentropy' loss because we have sparse labels (i.e., for each instance, there is just a target class index, from 0 to 9 in this case), and the classes are exclusive. If instead we had one target probability per class for each instance (such as one-hot vectors, e.g., [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.] to represent class 3), then we would need to use the 'categorical_crossentropy' loss instead. If we were doing binary classification or multilabel binary classification, then we would use the 'sigmoid' activation function in the output layer instead of the 'softmax' activation function, and we would use the 'binary_crossentropy' loss.

---------------------------------------------------------------------------------------------------

>>> history = model.fit(X_train, y_train, epochs=30,
...                     validation_data=(X_valid, y_valid))

We pass it the input features (X_train) and the target classes (y_train), as well as the number of epochs to train (or else it would default to just 1, which would definitely not be enough to converge to a good solution). We also pass a validation set (this is optional). Keras will measure the loss and the extra metrics on this set at the end of each epoch, which is very useful to see how well the model really performs. If the performance on the training set is much better than on the validation set, your model is probably overfitting the training set, or there is a bug, such as a data mismatch between the training set and the validation set.

---------------------------------------------------------------------------------------------------

In general you will get more bang for your buck by increasing the number of layers instead of the number of neurons per layer.

---------------------------------------------------------------------------------------------------

If you observe that the model is overfitting, you can increase the dropout rate. Conversely, you should try decreasing the dropout rate if the model underfits the training set. It can also help to increase the dropout rate for large layers, and reduce it for small ones. Moreover, many state-of-the-art architectures only use dropout after the last hidden layer, so you may want to try this if full dropout is too strong.

---------------------------------------------------------------------------------------------------

A common mistake is to use convolution kernels that are too large. For example, instead of using a convolutional layer with a 5 × 5 kernel, stack two layers with 3 × 3 kernels: it will use fewer parameters and require fewer computations, and it will usually perform better. One exception is for the first convolutional layer: it can typically have a large kernel (e.g., 5 × 5), usually with a stride of 2 or more. This will reduce the spatial dimension of the image without losing too much information, and since the input image only has three channels in general, it will not be too costly.

---------------------------------------------------------------------------------------------------
