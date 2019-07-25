import numpy as np
import pandas as pd

def parse_colors_config(colors_config):

    colors = pd.read_csv(colors_config, sep=' ')
    print(colors)

    
    Colors_config = {}

    for i in range(len(colors['colorname'])):
        Colors_config[colors['colorname'][i]] = [(colors['lowerH'][i], colors['lowerS'][i], colors['lowerV'][i]), (colors['upperH'][i], colors['upperS'][i], colors['upperV'][i])]

    return Colors_config

def detect_segment(mask):
    

    mask_xproj = [np.max(i) for i in mask.T]
       
    edges = []
    
    up = max(mask_xproj)
    down = np.min(mask_xproj) - np.max(mask_xproj)
    
    for i in range(1, len(mask_xproj)):
        
        if i == 1 and mask_xproj[i-1] == up:
            start = i
        if mask_xproj[i] - mask_xproj[i-1] == up:
            start = i
        if mask_xproj[i] - mask_xproj[i-1] == down:
            edges.append((start, i))
    Segments = []
    Segments_edges = []
    for edge in edges:
        
        V_segment = mask.T[edge[0]:edge[1]].T
        if len(V_segment[0]) == 0:
            continue
        current_segment = None
        for i in range(len(V_segment)):
            line = V_segment[i]
            
            if i == len(V_segment) - 1 and current_segment != None:
                Segments_edges.append((edge[0]-1, edge[1], left-1, i))
            
            if max(line) == 0:
                
                if current_segment == None:
                    continue
                Segments.append(current_segment)
                Segments_edges.append((edge[0]-1, edge[1], left-1, i))
                current_segment = None
                continue
                
            if current_segment == None:
                current_segment = [line]
                left = i
            
            else:
                current_segment.append(line)
    return Segments_edges
    

