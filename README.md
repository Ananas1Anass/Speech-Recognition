# Algorithms for Speech Recognition

Many algorithms are used to process speech and "correctly" convert speech into words.

For DL, Neural networks such as CNN, SNN, SRNN are used as models for the task, based on the output of MFCC on audio sample.

For Continuous Speech Recognition, difficulty rises because of the necessity to predict the upcoming word that follows the same meaning and logic of the sentence. 
Some algorithms are used such as  :

+ Viterbi
+ N-grams
+ HMM
+ MLP
+ PLP
+ Hybrid HMM-MLP

# CNN MODL

| CNN Layers | Description                                                                              |
|------------|------------------------------------------------------------------------------------------|
| 1          | Image input layer to adjust image dimensions                                             |
| 2          | To add filter size of pixels (padding equal 3)                                           |
| 3          | RelU layer (mean and deviation to 0)                                                     |
| 4          | Add pooling to reduce size with 3 stride and 2 padding                                   |
| 5          | 2*number of filter(numF) for padding [number of filter =10 ]                             |
| 6          | RelU layer and maxpooling2dlayer with stride 3 and padding 2(Batch normalization layer)  |
| 7          | Convolution 2D Layer (3,4*numF, ‘Padding’, ‘same')                                       |
| 8          | RelU layer (batch normalization)                                                         |
| 9          | Max Pooling 2D Layer([timePoolSize,1])                                                   |
| 10         | Dropout Layer (dropout rob), dropout, prevent overfitting                                |
| 11         | Fully Connected Layer (Number of Classes)                                                |
| 12         | Softmax Layer :  compute probability of each label                                       |
| 13         | Classification Layer, classify based on softmax, cost will be x-entropy                  |

Based on : Ayad Alsobhani  et al  2021  J. Phys.: Conf. Ser.  1973  012166


'''
pip install pydub
'''

