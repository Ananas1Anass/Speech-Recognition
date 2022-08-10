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

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 32, 32, 32)        160       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 16, 16, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 16, 16, 64)        8256      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 8, 8, 64)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 8, 8, 128)         32896     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 4, 4, 128)        0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 4, 4, 256)         131328    
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 2, 2, 256)        0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 2, 2, 1024)        1049600   
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 1, 1, 1024)       0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 1, 1, 1024)        4195328   
                                                                 
 flatten (Flatten)           (None, 1024)              0         
                                                                 
 dense (Dense)               (None, 1024)              1049600   
                                                                 
 dropout (Dropout)           (None, 1024)              0         
                                                                 
 dense_1 (Dense)             (None, 1024)              1049600   
                                                                 
 dropout_1 (Dropout)         (None, 1024)              0         
                                                                 
 dense_2 (Dense)             (None, 1024)              1049600   
                                                                 
 dropout_2 (Dropout)         (None, 1024)              0         
                                                                 
 dense_3 (Dense)             (None, 20)                20500     
                                                                 
=================================================================
Total params: 8,586,868
Trainable params: 8,586,868
Non-trainable params: 0
_________________________________________________________________

Based on : Ayad Alsobhani  et al  2021  J. Phys.: Conf. Ser.  1973  012166


'''
pip install pydub
'''

