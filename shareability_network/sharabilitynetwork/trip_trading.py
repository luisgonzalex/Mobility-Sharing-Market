import csv
import sys
import numpy
import networkx as nx
import pickle
import random


'''
performs trip trading algorithm as described in research doc
@:param dict() day_info - a dictionary containing "day", "delta_1" and "delta_2"
@:param float market_split - between 0-1, denoting the share of the market
@:param float rates - the rates being offered by
@:param matching_alg: choose from max_card_matching, max_weight_matching, max_profit_matching 
'''
def trip_trade(day_info, market_split, rates, matching_alg):
    delta_1 = day_info["delta_1"]
    delta_2 = day_info["delta_2"]
    day = day_info["day"]
    G = gen_graph(day, delta_1, delta_2)
    company1, company2 = split_graph(G, market_split)
    # how do initialize profits?
    # should technically be matching_alg here
    max_matching1 = nx.max_weight_matching(company1, weight='distance')
    max_matching2 = nx.max_weight_matching(company2, weight='distance')
    profits1 = calc_profits(company1, max_matching1, rates)
    profits2 = calc_profits(company2, max_matching2, rates)
    matched_trips1, matched_trips2 = [len(max_matching1)], [len(max_matching2)]
    profits_iter1, profits_iter2 = [profits1], [profits2]
    # add nodes that we need to visit
    # are trips that are not in the matching set
    # not matched and not previously offered?
    to_visit_1 = set()
    to_visit_2 = set()
    for node in company1.nodes:
        if not company1[node]:
            to_visit_1.add(node)
    for node in company2.nodes:
        if not company2[node]:
            to_visit_2.add(node)

    iters = 0
    while to_visit_1 and to_visit_2:
        company1, company2 = do_iteration(company1, company2, rates, to_visit_1)
        company1, company2 = do_iteration(company2, company1, rates, to_visit_2)
        max_matching1 = nx.max_weight_matching(company1, weight='distance')
        max_matching2 = nx.max_weight_matching(company2, weight='distance')
        matched_trips1.append(len(max_matching1))
        matched_trips2.append(len(max_matching2))
        profits1 = calc_profits(company1, max_matching1, rates)
        profits2 = calc_profits(company2, max_matching2, rates)
        profits_iter1.append(profits1)
        profits_iter2.append(profits2)
        iters += 1
    results = {}
    results['profits1'] = profits1
    results['profits2'] = profits2
    results['matchigs1'] = matched_trips1
    results['matchings2'] = matched_trips2
    results['iters'] = iters
    return results

def do_iteration(companyA, companyB, rates, queue):
    compAcopy = companyA.copy()
    compBcopy = companyB.copy()
    new_node = queue.pop()
    compAcopy.remove_node(new_node)
    # how do i calculate the price?
    est_price = rates['uberx']*companyA[new_node]['distance']
    compBcopy.add_node(new_node)
    new_max_matchingB = nx.max_weight_matching(companyB, weight='distance')
    # not sure how to calc new matching
    new_rev = rates['pool']
    if new_rev < est_price:
        return compAcopy, compBcopy
    return companyA, companyB


def calc_profits(graph, matchings, rates):
    # not sure if calculating profits correctly!!!
    profits = 0
    pool_rate_mile = rates['pool']['mile']
    pool_rate_min = rates['pool']['min']
    booking = rates['booking']
    driver_rate = rates['driver']
    min_fare = rates['min_fare']
    for edge1, edge2 in matchings:
        # 2 riders gain - 1 driver pay
        distance = graph[edge1][edge2]['distance']
        time = graph[edge1][edge2]['time']
        pool_profits = max(2*distance*pool_rate_mile+time*pool_rate_min+booking, min_fare) - driver_rate*distance
        profits += pool_profits
    return profits

'''
@:returns a tuple of graphs, split by split factor.
first graph has split_factor*graph nodes, second has the remainder
'''
def split_graph(G,split_factor):
    node_list = list(G.nodes)
    number_nodes_subgraph = int(len(node_list) * split_factor)
    sub_node_list_1 = []
    count = 0
    while count <= number_nodes_subgraph:
        a = random.randint(0,len(node_list) - 1)
        node = node_list[a]
        if node in sub_node_list_1:
            continue
        else:
            sub_node_list_1.append(node)
            count += 1
    #print(sub_node_list_1)
    sub_node_list_2 = list(set(node_list) - set(sub_node_list_1))
    #print(sub_node_list_2)

    sub_graph_1 = G.subgraph(sub_node_list_1).copy()
    sub_graph_2 = G.subgraph(sub_node_list_2).copy()
    # results1 = nx.max_weight_matching(sub_graph_1,weight='distance')
    # pickle.dump(results1, open("r2.p", "wb"))
    # results2 = nx.max_weight_matching(sub_graph_2,weight='distance')
    # pickle.dump(results2, open("r3.p", "wb"))
    return sub_graph_1, sub_graph_2

'''
generates a graph from the shareability network file
@:param int delta_1: the first delay
@:param int delta_2: the second delay
@:param int day: the day of the taxi_data
@:return a graph containing the 
'''
def gen_graph(day, delta_1, delta_2):
    raw_data = {}
    with open('P{}D{}D{}B{}.csv'.format(100, delta_1, delta_2, day)) as g:
        wt = csv.reader(g, delimiter=',')
        for row in wt:
            if row:
                raw_data[int(row[0])] = row[1:]

    node_list = []
    for node in raw_data:
        node_list.append(node)
        if len(raw_data[node]) == 0:
            continue
        index = int(len(raw_data[node]) / 4)
        for i in range(index):
            node_list.append(node + int(raw_data[node][4*i]))
    node_list = list(set(node_list))

    G = nx.Graph()
    for node in node_list:
        G.add_node(node)
    for node in raw_data:
        information = raw_data[node]
        index = int(len(raw_data[node]) / 4)
        for i in range(index):
            node_b = node + int(information[i*4])
            travel_time_save = int(information[i*4 + 2])
            travel_distance_save = int(information[i*4 + 3])
            G.add_edge(node, node_b, time=travel_time_save, distance=travel_distance_save)

    node_list = list(G.nodes)
    number_nodes_subgraph = int(len(node_list) * 0.01)
    print(number_nodes_subgraph)
    sub_node_list = []
    count = 0
    while count <= number_nodes_subgraph:
        a = random.randint(0, len(node_list) - 1)
        node = node_list[a]
        if node in sub_node_list:
            continue
        else:
            sub_node_list.append(node)
            count += 1

    sub_graph = G.subgraph(sub_node_list).copy()
    return sub_graph


if __name__ == '__main__':
    day_info = {}
    day_info["delta_1"] = 299
    day_info["delta_2"] = 300
    day_info["day"] = 55

