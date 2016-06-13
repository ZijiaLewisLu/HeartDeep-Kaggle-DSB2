# Working Log

## Building up net

- Issue in segmentation label
  - S>SoftmaxOutput compress output's last two dims, thus not right to label
  - S>Build own IOU Layer
    - tons of unknown….
    - test infer_shape
      - stupid error, causing type error

      - operation error, update mxnet and use ndarray

      - total elements not match?

        - R> in eval_metric, it compare prediction and label

        - however, I make prediction to be (1L,)

          ​
  - S>Using Logistic
    - required softmax_label not find
      - R> in iter, the name of label is set to softmax...
      - S> name the last layer as 'softmax' <居然tm好用？
    - NOT USE iter, pass numpy array in
      - Error: y must be two dim?!  WTF!!
    - Name laster layer 'softmax' works, but new error: Float point exception, core dumped … … 
      - subtract mean not working
  - S> Use Example
  - S> Use Minpy
  - S> Use quick_bind?

- Another common error:
  - number_epoch must be given, though doc says it's optional

- Floating point error:

  - Cause by UpSampling, don't know why
  - R>try replace it by deconv


- ISSUES in slice out of indice
  - possible reason: batch_size







