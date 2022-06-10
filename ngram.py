#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wedn Jun  8 11:30:25 2022

@author: anass
"""

import re

s ="Admirez le pouvoir insigne Et la noblesse de la ligne : Elle est la voix que la lumiere fit entendre Et dont parle Hermes Trismegiste en son Pimandre."

def generate_ngrams(s, n):
    s = s.lower()
    
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    
    tokens = [token for token in s.split(" ") if token != ""]
    
    # Generate n-grams
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


generate_ngrams(s,3)