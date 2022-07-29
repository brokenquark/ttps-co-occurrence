from cmath import nan
import gc
import re, networkx as nx
import pandas as pd
import domain
import tabulate as tb
from typing import Counter, List
from audioop import reverse
from email import header
from itertools import count
import pandas as pd 
import domain, utils
import tabulate as tb
from typing import List
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import networkx as nx 
import domain, utils
import statistics, math
import igraph
from networkx.algorithms import bipartite as bp
from networkx.algorithms import community as nxcm
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
# from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
from prefixspan import PrefixSpan
import functools
from itertools import count

def find_citation(text : str) -> str:
    tokens = text.split('(Citation: ')
    tokens = tokens[-1].strip().split(' ')
    outputText = ''
    for index in range(0, len(tokens)):
        outputText = outputText + ' ' + tokens[index]
    return outputText[0:-1].strip()

def findYear(text : str) -> str:
    regexPattern = r'''[0-9][0-9][0-9][0-9]'''
    match = re.search(regexPattern, text)
    return text[match.span()[0] : match.span()[1]]

def initializeTactics(tactics : List['domain.Tactic']) -> None:
    dfTactic = pd.read_excel('ttps.xlsx', sheet_name='tactic')
    for row in dfTactic.itertuples():
        tactic = domain.Tactic(row.tacticId, row.tacticName)
        
        if row.tacticId == 'TA0043' : tactic.sequence = 1
        if row.tacticId == 'TA0042' : tactic.sequence = 2
        if row.tacticId == 'TA0001' : tactic.sequence = 3
        if row.tacticId == 'TA0002' : tactic.sequence = 4
        if row.tacticId == 'TA0003' : tactic.sequence = 5
        if row.tacticId == 'TA0004' : tactic.sequence = 6
        if row.tacticId == 'TA0005' : tactic.sequence = 7
        if row.tacticId == 'TA0006' : tactic.sequence = 8
        if row.tacticId == 'TA0007' : tactic.sequence = 9
        if row.tacticId == 'TA0008' : tactic.sequence = 10
        if row.tacticId == 'TA0009' : tactic.sequence = 11
        if row.tacticId == 'TA0011' : tactic.sequence = 12
        if row.tacticId == 'TA0010' : tactic.sequence = 13
        if row.tacticId == 'TA0040' : tactic.sequence = 14
        
        
        tactics.append(tactic)

def initializeTechniques(techniques : List['domain.Technique']) -> None:
    dfTechnique = pd.read_excel('ttps.xlsx', sheet_name='technique')
    for row in dfTechnique.itertuples():
        ifAny = [x for x in techniques if x.id == row.techniqueId]
        if len(ifAny) == 0:
            technique = domain.Technique(row.techniqueId, row.techniqueName)
            techniques.append(technique)

def initializeTacticTechniqueMapping(tactics : List['domain.Tactic'], techniques : List['domain.Technique']) -> None:
    dfTechnique = pd.read_excel('ttps.xlsx', sheet_name='technique')
    for row in dfTechnique.itertuples():
        technique = [x for x in techniques if x.id == row.techniqueId][0]
        tactic = [x for x in tactics if x.id == row.tactics][0]
        if tactic not in technique.tactics: technique.tactics.append(tactic)
        if technique not in tactic.techniques: tactic.techniques.append(technique)

def initializeProcedures(procedures : List['domain.Procedure'], techniques : List['domain.Technique']) -> None:
    dfProcedures = pd.read_excel('technique.xlsx', sheet_name='procedure')
    dfProcedures = dfProcedures[['sourceId', 'targetId', 'citation']]
    dfProcedures['targetId'] = dfProcedures['targetId'].apply(lambda row : row[0:5])
    dfProcedures['citation'] = dfProcedures['citation'].apply(find_citation)
    dfPRef = pd.read_excel('technique.xlsx', sheet_name='citations')
    dfPRef['reference'] = dfPRef['reference'].apply(findYear)
    dfPRef = dfPRef.drop_duplicates()
    dfm = pd.merge(dfProcedures, dfPRef, how='left', left_on=['citation'], right_on=['citation'])
    dfm = dfm.drop_duplicates()
    dfm = dfm.dropna()

    for row in dfm.itertuples():
        procedure = domain.Procedure(row.sourceId + ':' + row.targetId + ':' + '-'.join(str(row.citation).split(' ')))
        procedure.technique = next( (x for x in techniques if x.id == row.targetId), None)
        procedure.year = row.reference
        procedure.name = row.sourceId + ':' + row.targetId
        procedure.reference = str(row.citation)
        if 'G' in procedure.id: 
            procedure.type = 'group'
        else: 
            procedure.type = 'software'
        procedures.append(procedure)

def initializeGroups(groups : List['domain.Group'], techniques : List['domain.Technique']) -> None:
    dfGroups = pd.read_excel('groups.xlsx', sheet_name='ttps')
    dfGroups = dfGroups[['sourceId', 'targetId']]
    dfGroups['targetId'] = dfGroups['targetId'].apply(lambda v : v[0:5])
    dfGroups = dfGroups.drop_duplicates()
    dfg = dfGroups.groupby(['sourceId'])
    
    for name, group in dfg:
        g = domain.Group(name)
        
        for row in group.itertuples():
            ttp = [x for x in techniques if x.id == row.targetId][0]
            if ttp not in g.techniques: 
                g.techniques.append(ttp)
        groups.append(g)

def initializeSoftwares(softwares : List['domain.Software'], techniques : List['domain.Technique']) -> None:
    dfSoftwares = pd.read_excel('software.xlsx', sheet_name='ttps')
    dfSoftwares = dfSoftwares[['sourceId', 'targetId']]
    dfSoftwares['targetId'] = dfSoftwares['targetId'].apply(lambda v : v[0:5])
    dfSoftwares = dfSoftwares.drop_duplicates()
    dfg = dfSoftwares.groupby(['sourceId'])
    
    for name, group in dfg:
        software = domain.Software(name)
        
        for row in group.itertuples():
            ttp = [x for x in techniques if x.id == row.targetId][0]
            if ttp not in software.techniques: 
                software.techniques.append(ttp)
        softwares.append(software)

def buildDataSchema(tactics : List['domain.Tactic'], techniques : List['domain.Technique'], procedures : List['domain.Procedure'], groups : List['domain.Group'], softwares : List['domain.Software']) -> None:
    initializeTactics(tactics)
    initializeTechniques(techniques)
    initializeTacticTechniqueMapping(tactics, techniques)
    initializeProcedures(procedures, techniques)
    initializeGroups(groups, techniques)
    initializeSoftwares(softwares, techniques)
    return

def degree_centrality(graph):
    max = 0
    for edge in graph.edges:
        if max < graph.edges[edge]['count']:
            max = graph.edges[edge]['count']

    dc = {}
    for node in graph.nodes:
        total = 0
        for item in graph.neighbors(node):
            total += graph.edges[node, item]['count']
        dc[f'{node}'] = total/(max*len(graph))
    return dc

def initializeCocGraph(groupsList : List['domain.Group'], softwareList : List['domain.Software'], cocTTPs : List[List['domain.Technique']], techniques : List['domain.Technique'], tactics : List['domain.Tactic'], min_cooccurring_technique = 3, min_pct_cooccurrence = 5) -> nx.Graph:
    allTechniques = []

    for g in groupsList:
        for te in g.techniques:
            allTechniques.append(te)

    allTechniques = list( set(allTechniques) )

    for s in softwareList:
        for te in s.techniques:
            allTechniques.append(te)

    allTechniques = list( set(allTechniques) )
    allTechniques.sort(key=lambda t : t.id)

    # cocTTPs = []
    cocTTPs.extend([g.techniques for g in groupsList if len(g.techniques) >= min_cooccurring_technique])
    cocTTPs.extend([s.techniques for s in softwareList if len(s.techniques) >= min_cooccurring_technique])

    ttpsTuples = []

    for ttp1 in allTechniques:
        for ttp2 in allTechniques:
            count = 0
            for item in cocTTPs:
                if ttp1 in item and ttp2 in item and ttp1 != ttp2:
                    count += 1
            if (ttp1, ttp2, count) not in ttpsTuples and (ttp2, ttp1, count) not in ttpsTuples and ttp1 != ttp2 and count > len(cocTTPs)*min_pct_cooccurrence/100:
                ttpsTuples.append((ttp1, ttp2, count))

    ttpsTuples.sort(key= lambda i : i[2], reverse=True)
    
    graph = nx.Graph()
    graph.add_nodes_from([x.id for x in allTechniques])
        
    for node in graph.nodes:
        graph.nodes[node]['tactic'] = next( (x.tactics[0].id for x in techniques if x.id == node) )
        te = next( (x for x in techniques if x.id == node) )
        graph.nodes[node]['frequency'] = len([x for x in cocTTPs if te in x])
        # print(graph.nodes[node])

    for item in ttpsTuples:
        graph.add_edge(item[0].id, item[1].id, count = item[2], distance = ttpsTuples[0][2] + 1 - item[2])
    
    return graph

def generateRules(cocTTPs : List[List['domain.Technique']]):
    transactions = []

    for cases in cocTTPs:
        transaction = []
        transaction.extend( [x.id for x in cases] )
        transactions.append(transaction)

    # print(transactions)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    # print(df.head())

    frequent_itemsets = fpgrowth(df, min_support=0.10, use_colnames=True)
    frequent_itemsets['len'] = frequent_itemsets['itemsets'].apply(lambda x : len(x))
    
    dflen = frequent_itemsets.query("len == 2")
    
    ttt = [list(x)[0] for x in dflen['itemsets'].tolist()]
    print(f'=========={len(set(ttt))}')
    
    # print(tb.tabulate(dflen.sort_values('support', ascending=False).head(100000), headers='keys', tablefmt='psql'))
    # print(tb.tabulate(dflen.describe(), headers='keys', tablefmt='psql'))
    
    lengths = []
    for item in frequent_itemsets.itertuples():
        lengths.append(len(item.itemsets))
    # print(Counter(lengths))

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.505)
    # rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
    print(f'*** rules ***')
    # print(tb.tabulate(rules.sort_values('confidence', ascending=False).head(20), headers='keys', tablefmt='psql'))
    # print(tb.tabulate(rules.sort_values('lift', ascending=False).head(20), headers='keys', tablefmt='psql'))
    # print(tb.tabulate(rules, headers='keys', tablefmt='psql'))
    print(len(rules))
    
    rules['alen'] = rules['antecedents'].apply(lambda x : len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x : len(x)) 

    dfq = rules.query("alen + clen == 2")
    # item = dfq.loc[1, 'antecedents']
    # print(list(item)[0])
    # print(tb.tabulate(dfq.sort_values(by='confidence', ascending=False).head(20), headers='keys', tablefmt='psql'))
    # cofValues = dfq['confidence'].tolist()
    
    # print(f'****** {len(dfq)} {min(cofValues)} {max(cofValues)} {statistics.mean(cofValues)} {statistics.quantiles(cofValues, n=4)}')
    
    
    
    
    return rules

def generateRulesTactic(cocTTPs : List[List['domain.Technique']]):
    transactions = []

    for cases in cocTTPs:
        transaction = []
        transaction.extend( [x.tactics[0].id for x in cases] )
        transaction = list(set(transaction))
        transactions.append(transaction)

    # print(transactions)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    # print(df.head())

    frequent_itemsets = fpgrowth(df, min_support=0.10, use_colnames=True)
    frequent_itemsets['len'] = frequent_itemsets['itemsets'].apply(lambda x : len(x))
    
    dflen = frequent_itemsets.query("len == 1")
    
    print(tb.tabulate(dflen.sort_values('support', ascending=False), headers='keys', tablefmt='psql'))
    print(len(frequent_itemsets))
    
    lengths = []
    for item in frequent_itemsets.itertuples():
        lengths.append(len(item.itemsets))
    print(Counter(lengths))

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.50)
    # rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.75)
    print(f'*** rules ***')
    # print(tb.tabulate(rules.sort_values('support', ascending=False), headers='keys', tablefmt='psql'))
    # print(tb.tabulate(rules, headers='keys', tablefmt='psql'))
    # print(rules.dtypes)
    return rules


def getTechniqueFrequentSequence(cocTTPs : List[List['domain.Technique']], techniques : List['domain.Technique'], tactics : List['domain.Tactic'] ):
    print(f'*** frequent sequence of techniques ***')
    transactions = []

    for cases in cocTTPs:
        transaction = []
        sortedTTPs = sorted(cases, key = lambda t : t.tactics[0].sequence )
        transaction.extend( [x.name for x in sortedTTPs] )
        transactions.append(transaction)


    ps = PrefixSpan(transactions)
    tactics.sort(key=lambda ta : ta.sequence)


    for item in ps.topk(100):
        if len(item[1]) > 2:
            text = ''
            for element in item[1]:
                te = next( (x for x in techniques if x.name == element) )
                ta = next( (x for x in tactics if te in x.techniques) )
                text += f'{te.name}@{ta.name} -->'
            print(text)
    return


def generateGraph(cocTTPs : List[List['domain.Technique']], techniques : List['domain.Technique'], tactics : List['domain.Tactic']) -> nx.Graph:

    df = generateRules(cocTTPs)
    df['alen'] = df['antecedents'].apply(lambda x : len(x)) 
    df['clen'] = df['consequents'].apply(lambda x : len(x)) 
    dfq = df.query("alen == 1 and clen == 1")
    
    ttpsTuples = []

    for row in dfq.itertuples():
        ttpsTuples.append([( list(row.antecedents)[0], list(row.consequents)[0]  ), row.confidence, row.support])

    techniqueNames = [x.id for x in techniques]

    edges = []
    
    for item in ttpsTuples:
        edges.append([(item[0][0], item[0][1]), item[1], item[2]])
              

    cocGraph = nx.Graph()

    for item in edges:
        if item[0][0] not in list(cocGraph.nodes):
            cocGraph.add_node(item[0][0])
        if item[0][1] not in list(cocGraph.nodes):
            cocGraph.add_node(item[0][1])
        
        cocGraph.add_edge(item[0][0], item[0][1], weight = item[1], count = len(cocTTPs) * item[2], distance = 1 - item[1])
    
    for node in cocGraph.nodes():
        cocGraph.nodes[node]['tactic'] = next( (x.tactics[0].id for x in techniques if x.id == node) )
        te = next( (x for x in techniques if x.id == node) )
        cocGraph.nodes[node]['frequency'] = len([x for x in cocTTPs if te in x])
        cocGraph.nodes[node]['title'] = f'{cocGraph.nodes[node]["tactic"]}:{node}'
    
    print(f'number of nodes: {len(cocGraph.nodes)}')
    print(f'number of edges: {len(cocGraph.edges)}')
    
    cocGraph = nx.minimum_spanning_tree(cocGraph, weight='distance')
    
    print(f'number of nodes: {len(cocGraph.nodes)}')
    print(f'number of edges: {len(cocGraph.edges)}')
    
    plt.close()
    nx.draw_spring(cocGraph, with_labels=True, node_color='gold')
    plt.show()
    
    return cocGraph

def generateDiGraph2(cocTTPs : List[List['domain.Technique']], techniques : List['domain.Technique'], tactics : List['domain.Tactic']) -> nx.DiGraph:

    df = generateRules(cocTTPs)
    # getTechniqueFrequentSequence(cocTTPs, techniques, tactics)
    
    # print(df)
    # print(df.loc[0, 'antecedents'])
    df['alen'] = df['antecedents'].apply(lambda x : len(x)) 
    df['clen'] = df['consequents'].apply(lambda x : len(x)) 

    dfq = df.query("alen == 1 and clen == 1")
    # item = dfq.loc[1, 'antecedents']
    # print(list(item)[0])
    # print(tb.tabulate(dfq, headers='keys', tablefmt='psql'))
    cofValues = dfq['confidence'].tolist()
    
    print(f'****** {statistics.quantiles(cofValues, n=4)}')

    dfqq = df.query("alen == 2 and clen == 1")
    # print(f'dfqq ==> {len(dfqq)}')
    # print(df.describe())
    
    
    ttpsTuples = []

    for row in dfq.itertuples():
        ttpsTuples.append([( list(row.antecedents)[0], list(row.consequents)[0]  ), row.confidence, row.support])

    techniqueNames = [x.id for x in techniques]

    edges = []
    
    for item in ttpsTuples:
        edges.append([(item[0][0], item[0][1]), item[1], item[2]])

    # for t1 in techniqueNames:
    #     for t2 in techniqueNames:
    #         if len([x for x in edges if (t1,t2) == x[0] or (t2,t1) == x[0] ]) == 0:
    #             pair1 = next( (x for x in ttpsTuples if (t1,t2) == x[0] ), None )
    #             pair2 = next( (x for x in ttpsTuples if (t2,t1) == x[0] ), None )
                
    #             if pair1 == None and pair2 == None:
    #                 continue
                
    #             if pair1 != None and pair2 == None:
    #                 edges.append( [(t1, t2), pair1[1]] )
                
    #             if pair1 == None and pair2 != None:
    #                 edges.append( [(t2, t1), pair2[1]] )
                    
    #             if pair1 != None and pair2 != None:
    #                 if pair1[1] > pair2[1] : 
    #                     edges.append( [(t1, t2), pair1[1]] )
    #                 else:
    #                     edges.append( [(t2, t1), pair2[1]] )
                    

    cocDiGraph = nx.DiGraph()

    for item in edges:
        if item[0][0] not in list(cocDiGraph.nodes):
            cocDiGraph.add_node(item[0][0])
        if item[0][1] not in list(cocDiGraph.nodes):
            cocDiGraph.add_node(item[0][1])
        
        cocDiGraph.add_edge(item[0][0], item[0][1], weight = item[1], count = len(cocTTPs) * item[2], distance = 1 - item[1])
    
    alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]
    idx = 0
    for node in cocDiGraph.nodes():
        cocDiGraph.nodes[node]['tactic'] = next( (x.tactics[0].id for x in techniques if x.id == node) )
        te = next( (x for x in techniques if x.id == node) )
        cocDiGraph.nodes[node]['frequency'] = len([x for x in cocTTPs if te in x])
        cocDiGraph.nodes[node]['title'] = f'{cocDiGraph.nodes[node]["tactic"]}:{node}'
        # cocDiGraph.nodes[node]['code'] = f'{alph[idx]}'
        # idx += 1
    
    ig = igraph.Graph.from_networkx(cocDiGraph)
    edges = ig.feedback_arc_set()

    tuples = []
    for id in edges:
        source = ig.vs[ig.es[id].source]['_nx_name']
        target = ig.vs[ig.es[id].target]['_nx_name']
        tuples.append((source,target))

    # print(tuples)
        
    # print(ig.es[0].source)
    # print(ig.es[0].target)
    # # print(ig.vs)
    # # print(ig.get_edgelist())
    # # print(edges)

    # cocDiGraph2 = cocDiGraph.copy()

    # for e in tuples:
    #     cocDiGraph.remove_edge(e[0], e[1])

    # nodes = [n for n in cocDiGraph.nodes(data=False)]
    # for n1 in nodes:
    #     for n2 in nodes:
    #         if cocDiGraph.has_edge(n1, n2) and cocDiGraph.has_edge(n2, n1):
    #             # print(f'{cocDiGraph.edges[n1,n2]} *** {cocDiGraph.edges[n2,n1]}')
    #             e1 = cocDiGraph.edges[n1,n2]
    #             e2 = cocDiGraph.edges[n2,n1]
    #             if e1['weight'] > e2['weight']:
    #                 cocDiGraph.remove_edge(n2, n1)
    #             else:
    #                 cocDiGraph.remove_edge(n1, n2)
    
    print(f'number of nodes: {len(cocDiGraph.nodes)}')
    print(f'number of edges: {len(cocDiGraph.edges)}')
    # print(f'density: {nx.density(cocDiGraph)}')
    # print(f'diameter: {nx.diameter(cocDiGraph.to_undirected())}')
    # print(f'radius: {nx.radius(cocDiGraph.to_undirected())}')
    # print(f'eccentricity: {nx.eccentricity(cocDiGraph.to_undirected())}')
    
    # gcenter = nx.center(cocDiGraph.to_undirected())
    # print([x for x in gcenter])
    
    # for node in cocDiGraph.nodes(data=True):
    #     print(node[1]['tactic])
    
    # for edge in cocDiGraph.edges(data=True):
    #     print(edge)
    
    
    # dg = nx.DiGraph()
    # dg.add_node('a')
    # dg.add_node('b')
    # dg.add_edge('a', 'b')
    
    tacticgroups = list(set(nx.get_node_attributes(cocDiGraph,'tactic').values()))
    plt.figure(3,figsize=(12,8))
    pos = nx.circular_layout(cocDiGraph)
    colors = ['yellow', 'orange', 'cyan', 'gold', 'magenta', 'pink', 'lime']
    shapes = ['d', 'X', 'P']
    shapes = ['o', 'o', 'o']
    tacticnames = []
    for item in tacticgroups:
        tacticnames.append(next( (x.name for x in tactics if x.id == item) ))
    
    labels = [n[1]['title'] for n in cocDiGraph.nodes(data=True)]
    labels = {n[0]: n[1]['title'] for n in cocDiGraph.nodes(data=True)}
    # labels = {n[0]: n[1]['code'] for n in cocDiGraph.nodes(data=True)}
    tacticNameLists = [n[1]['tactic'] for n in cocDiGraph.nodes(data=True)]
    print(tacticNameLists)
    print(Counter(tacticNameLists))
    print({n: n for n in cocDiGraph})
    
    
    for index in range(0, len(tacticgroups)):
        # print(tacticgroups[index])
        # print(colors[index])
        searchNodes = [x[0] for x in cocDiGraph.nodes(data=True) if x[1]['tactic'] == tacticgroups[index]]
        nsizes = [cocDiGraph.nodes[x]['frequency']*2000/669 for x in searchNodes]
        # print(nsizes)
        # nx.draw_networkx_nodes(cocDiGraph, pos=pos, nodelist=searchNodes, node_size=150, alpha=0.99, node_color=colors[index % 7], node_shape=shapes[index % 3], label=tacticnames[index])
    
    
    esizes = []
    for edge in cocDiGraph.edges:
        # print(f'{cocDiGraph.edges[edge[0], edge[1]]["weight"]}')
        esizes.append(cocDiGraph.edges[edge[0], edge[1]]["weight"])
    
    # , connectionstyle="arc3,rad=0.4"
    nx.draw_networkx_edges(cocDiGraph, pos=pos, width=0.3, edge_color='grey')
    # labels=labels,
    nx.draw_networkx_labels(cocDiGraph,  pos=pos, font_color='blue', font_size=15, font_weight='bold')
    
    plt.legend(scatterpoints = 1)
    # plt.show()
    
    # searchNodes = [x for x in cocDiGraph.nodes(data=True) if x[1]['tactic'] == 'TA0005']
    # print(searchNodes)
    
    # nx.draw_circular(cocDiGraph, with_labels=True)
    # nx.draw_kamada_kawai(cocDiGraph, with_labels=False)
    # plt.show()

    # print(len(cocDiGraph.nodes))
    ig = igraph.Graph.from_networkx(cocDiGraph)
    # layout = ig.layout("kk")
    # igraph.plot(ig, layout=layout)
    # plt.show()

    # # out_fig_name = "digraph.eps"
    # visual_style = {}
    # colours = ['#fecc5c', '#a31a1c']
    # visual_style["bbox"] = (3000,3000)
    # visual_style["margin"] = 17
    # visual_style["vertex_color"] = 'red'
    # visual_style["vertex_size"] = 50
    # visual_style["vertex_label_size"] = 80
    # visual_style["edge_curved"] = False
    # my_layout = ig.layout_auto()
    # visual_style["layout"] = my_layout
    # # igraph.plot(ig, out_fig_name, **visual_style)
    # igraph.plot(ig, **visual_style)
    
    return cocDiGraph

def normalizeList(values):
    minVal = min(values)
    maxVal = max(values)
    
    newValues = [ (x - minVal)/(maxVal + 0.0000000001 - minVal) for x in values]
    return newValues