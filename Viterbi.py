#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 09 14:58:42 2022

@author: anass
"""

# %% revoir Natural Language Toolkit


import numpy as np
import operator

# %% NLTK tag

"""
 transition probabilities = P(VP | NP) (example)
 conditional_probabilities = P(Anass | NP)
"""

def viterbi (transition_probabilities, conditional_probabilities):
    # Initialisation tout
    num_samples = conditional_probabilities.shape[1]
    num_states = transition_probabilities.shape[0] # number of states

    c = np.zeros(num_samples) #scale factors 
    viterbi = np.zeros((num_states,num_samples)) # initialise viterbi table
    best_path_table = np.zeros((num_states,num_samples)) # initialise the best path table
    best_path = np.zeros(num_samples).astype(np.int32) # this will be your output

    # initial values for viterbi and best path 
    viterbi[:,0] = conditional_probabilities[:,0]
    c[0] = 1.0/np.sum(viterbi[:,0])
    viterbi[:,0] = c[0] * viterbi[:,0] # apply the scaling factor

    # C- Iterations
    for t in range(1, num_samples):
        for s in range (0,num_states): 
            trans_p = viterbi[:, t-1] * transition_probabilities[:,s] # transition probs of each state transitioning
            best_path_table[s,t], viterbi[s,t] = max(enumerate(trans_p), key=operator.itemgetter(1))
            viterbi[s,t] = viterbi[s,t] * conditional_probabilities[s][t]

        c[t] = 1.0/np.sum(viterbi[:,t]) # scaling factor
        viterbi[:,t] = c[t] * viterbi[:,t]

    ## D - Back-tracking
    best_path[num_samples-1] =  viterbi[:,num_samples-1].argmax() # last state
    for t in range(num_samples-1,0,-1): 
        best_path[t-1] = best_path_table[best_path[t],t]
    return best_path