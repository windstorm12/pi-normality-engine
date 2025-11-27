#!/usr/bin/env python3
"""
If π has hidden patterns, it should compress differently than random digits.
We'll compare compression ratios at different scales and positions.
"""

import zlib
import lzma
import bz2
from collections import defaultdict

def compression_anomaly_scan(pi_digits, window_size=10000, stride=1000):
    """
    Look for regions that compress unusually well or poorly
    """
    anomalies = []
    
    for start in range(0, len(pi_digits) - window_size, stride):
        chunk = pi_digits[start:start + window_size]
        
        # Try multiple compression algorithms
        original_size = len(chunk)
        zlib_size = len(zlib.compress(chunk.encode()))
        lzma_size = len(lzma.compress(chunk.encode()))
        bz2_size = len(bz2.compress(chunk.encode()))
        
        # Compare to random baseline
        import random
        random_chunk = ''.join([str(random.randint(0,9)) for _ in range(window_size)])
        random_zlib = len(zlib.compress(random_chunk.encode()))
        
        compression_ratio = zlib_size / original_size
        random_ratio = random_zlib / original_size
        
        # Anomaly score: how different from random?
        anomaly_score = abs(compression_ratio - random_ratio) / random_ratio
        
        if anomaly_score > 0.05:  # 5% deviation
            anomalies.append({
                'position': start,
                'score': anomaly_score,
                'ratio': compression_ratio,
                'algorithms': {
                    'zlib': zlib_size/original_size,
                    'lzma': lzma_size/original_size,
                    'bz2': bz2_size/original_size
                }
            })
    
    return anomalies