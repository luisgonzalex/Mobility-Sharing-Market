#!/usr/bin/env python
# coding: utf-8
# In[4]:
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
import math

url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2014-01.csv'
fn = 'yellow_tripdata_2014-01.csv'
map_fn = 'NYC Street Centerline (CSCL).geojson'
with open(map_fn) as f:
    map_data = json.load(f)

def create_graph(map_data):
    ''''given a list of map data, this function returns a dictionary mapping
        from intersections (coordinates (long,lat)) to other intersections.
        '''
    loc_to_stname = {}  # maps coord to street name
    st_to_st = {}          # maps st name to other st's
    street_names = set()  # to check how many distinct street names we have
    st_name_obj = {}    # maps street name to json object
    xing_to_xing = {}   # this will be our real graph, it maps an intersection to other intersections that link
    graph = {}

    # populate the crossing to crossing map
    for entry in map_data:
        intersections = entry['geometry']['coordinates'][0]
        if len(intersections) > 1:
            add_to_graph(xing_to_xing, intersections, entry['properties']['trafdir'])

    intersection_freq, intersection_freq_map = count_freq(xing_to_xing)
    print("format is k:v, where k is an intersection of k streets and "
          "v is the number of such intersections in manhattan\n", intersection_freq)
    # intersections that only map to one intersection are likely not intersections
    # but rather road segments, so let's get rid of them
    for k in intersection_freq_map[1]:
        xing_to_xing.pop(k)

    # first we map intersections to streets that intersect at those coords
    for entry in map_data:
        street_name = entry['properties']['st_label']
        street_names.add(street_name)
        intersections = entry['geometry']['coordinates'][0]
        st_name_obj[street_name] = entry
        for intersection in intersections:
            # intersection = [round(elt, 6) for elt in intersection]
            loc_to_stname[tuple(intersection)] = loc_to_stname.get(tuple(intersection), set()).union({street_name})

    # freq used for debugging (maps k:v where k is number of streets with same coord and v is number of occurrences
    # ht maps k to list of intersections of k streets
    freq, ht = count_freq(loc_to_stname)

    # we have a lot of intersections that found no pair. These are typically intersections that lead to
    # allys and such. lets get rid of them.
    for k in ht[1]:
        loc_to_stname.pop(k)

    for entry in map_data:
        street_name = entry['properties']['st_label']
        intersections = entry['geometry']['coordinates'][0]
        for intersection in intersections:
            # intersection = [round(elt, 5) for elt in intersection]
            graph[street_name] = graph.get(street_name, set()).union(loc_to_stname.get(tuple(intersection), set()))
        graph[street_name].discard(street_name)

    return xing_to_xing

def add_to_graph(graph, intersections, code):
    ''' given a graph, it adds the intersection data, data, and uses a code to
        determine if we need to add the reverse (undirected edges) or if we need to do this backwards '''
    if code in {"TW", "TF"}:
        add_to_graph(graph, intersections[::-1], "FT")
    if code in {"TW", "FT"}:
        for i in range(1, len(intersections)):
            crossing1 = tuple(intersections[i-1])
            crossing2 = tuple(intersections[i])
            graph[crossing1] = graph.get(crossing1, set()).union({crossing2})

def count_freq(ht):
    ''' given a dictionary, returns two dictionaries
        freq maps len(val) : instances of this len
        freq_obj maps len(val): '''
    freq = {}
    freq_obj = {}
    for k,v in ht.items():
        freq[len(v)] = freq.get(len(v),0) + 1
        freq_obj[len(v)] = freq_obj.get(len(v), []) + [k]
    return freq, freq_obj

locations = map_data['features']

# manhattan_data = [elt for elt in locations if elt['properties']['borocode'] == '1']

manhattan_data = [elt for elt in locations if
                  elt['properties']['borocode'] == '1' and
                  elt['properties']['rw_type'] in {str(elt) for elt in [1,2,3,4]} and
                  elt['properties']['trafdir'] in {'FT', 'TW', 'TF'}]

print("number of valid streets in Manhattan based on data: \n", len(manhattan_data))
# we create a mapping from coordinates to other intersections that are accessible from that coordinate
mapping = create_graph(manhattan_data)

print("after cleaning up data to remove non-intersections: \n", count_freq(mapping)[0])
print("total intersections considered \n", len(mapping))
# for k,v in mapping.items():
#     print(k, v)
# taxi_data = pd.read_pickle('taxi_data_2014.pkl')
#
# taxi_data[" pickup_datetime"] = pd.to_datetime(taxi_data[" pickup_datetime"], infer_datetime_format=True)
#
# # we should sort the taxi data by date, as we likely need this for our time estimation algorithm
# taxi_data = taxi_data.sort_values(" pickup_datetime")
#
# # lets do 10% of traffic in January 10, from 5AM-10PM, for example
# date1 = '2014-01-10 08:00:00'
# date2 = '2014-01-10 23:00:00'
# small_data = taxi_data[(taxi_data[' pickup_datetime'] > date1) & (taxi_data[' pickup_datetime'] <= date2)]
# small_data = small_data[:len(small_data)//10]
# small_data_list = small_data["vendor_id"]
# print(small_data.head())



