from ast import Lambda
from audioop import reverse
from cProfile import label
from ctypes import util
from email import header
from itertools import count
from platform import node
import this
import pandas as pd 
import domain, utils
import tabulate as tb
from typing import Counter, List
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
from scipy.stats import pointbiserialr


tactics : List['domain.Tactic'] = []
techniques : List['domain.Technique'] = []
procedures  : List['domain.Procedure'] = []
groups  : List['domain.Group'] = []
softwares  : List['domain.Software'] = []
cocTTPs : List[List['domain.Technique']] = []

utils.buildDataSchema(tactics, techniques, procedures, groups, softwares)
cocGraph : nx.Graph = utils.initializeCocGraph(groups, softwares, cocTTPs, techniques, tactics)
cocDiGraph : nx.DiGraph =  utils.generateDiGraph2(cocTTPs, techniques, tactics)

# *** find top occurring techniques (Table III)
def findTopTenTechniques(techniques, cocTTPs):
    techniquesSortedBySupport = []
    for te in techniques:
        sum = 0
        for coc in cocTTPs:
            if te in coc: 
                sum += 1
        techniquesSortedBySupport.append((te, sum))

    techniquesSortedBySupport.sort(key = lambda v : v[1], reverse=True)

    idx = 0
    for item in techniquesSortedBySupport[0:10]:
        idx += 1
        print(f"[{idx}] {item[0].id}: {item[0].name} @ {item[0].tactics[0].id}: {item[0].tactics[0].name} ==> {item[1]/len(cocTTPs)}")
    return

# *** find top tactics (Table IV)
def findTopTactics(techniques, tactics, cocTTPs):
    techniquesSortedBySupport = []
    for te in techniques:
        sum = 0
        for coc in cocTTPs:
            if te in coc: 
                sum += 1
        techniquesSortedBySupport.append((te, sum))

    techniquesSortedBySupport.sort(key = lambda v : v[1], reverse=True)

    idx = 0
    for item in techniquesSortedBySupport[0:10]:
        idx += 1
        print(f"[{idx}] {item[0].id}: {item[0].name} @ {item[0].tactics[0].id}: {item[0].tactics[0].name} ==> {item[1]/len(cocTTPs)}")
    
    
    topFourteenTechniques = []
    topTechniquesWithMinSpprt = [x[0] for x in techniquesSortedBySupport if x[1] > 59.9]

    columns = ['support', 'count', 'min', 'avg', 'med', 'stdev', 'max', 'top']
    index = [ta.id + ': ' + ta.name for ta in tactics]
    df = pd.DataFrame(index=index, columns=columns)

    for ta in tactics:
        topTechInThisTa = None
        maxValue = -1
        values = []
        support = 0
        for coc in cocTTPs:
            if ta in [x.tactics[0] for x in coc]:
                support += 1
        for te in topTechniquesWithMinSpprt:
            if te.tactics[0] == ta:
                value = [x[1] for x in techniquesSortedBySupport if x[0] == te][0]
                values.append(value/len(cocTTPs))
                if value > maxValue:
                    topTechInThisTa = te
                    maxValue = value
        if len(values) > 0:
            if len(values) > 1:
                topFourteenTechniques.append(topTechInThisTa)
                df.loc[f"{ta.id}: {ta.name}"] = [(support/len(cocTTPs)), (len(values)), (min(values)), (statistics.mean(values)), (statistics.median(values)), (statistics.stdev(values)), (max(values)), (topTechInThisTa.id + ': ' + topTechInThisTa.name)]
            if len(values) == 1:
                topFourteenTechniques.append(topTechInThisTa)
                df.loc[f"{ta.id}: {ta.name}"] = [(support/len(cocTTPs)), (len(values)), (min(values)), (statistics.mean(values)), (statistics.median(values)), (0), (max(values)), (topTechInThisTa.id + ': ' + topTechInThisTa.name)]

    for cols in columns[:-1]:
        df[cols] = df[cols].astype(float)
    df = df.round(2)
    print(tb.tabulate(df.sort_values(by='support', ascending=False), headers='keys', tablefmt='psql'))
    return

# top ten combinations (Table V)
def getTopTenCombinations(cocTTPs):
    rules = utils.generateRules(cocTTPs)

    rules['alen'] = rules['antecedents'].apply(lambda x : len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x : len(x)) 
    dfq = rules.query("alen + clen == 2")

    print(dfq[['antecedents', 'consequents', 'support', 'confidence']].sort_values(by='support', ascending=False).head(20))
    return

# get top ten simple rules (Table VI)
def getTopTenSimpleRules(cocTTPs):
    rules = utils.generateRules(cocTTPs)

    rules['alen'] = rules['antecedents'].apply(lambda x : len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x : len(x)) 
    dfq = rules.query("alen + clen == 2")

    print(dfq[['antecedents', 'consequents', 'support', 'confidence']].sort_values(by='confidence', ascending=False).head(20))
    return

# get top ten compound rules (Table VII)
def getTopTenCompoundRules(cocTTPs):
    rules = utils.generateRules(cocTTPs)

    rules['alen'] = rules['antecedents'].apply(lambda x : len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x : len(x)) 
    dfq = rules.query("alen + clen > 2")

    print(dfq[['antecedents', 'consequents', 'support', 'confidence']].sort_values(by='confidence', ascending=False).head(20))
    return

# get the adjacency matrix of the co-occurrence network (table VIII)
def getAdjacencyMatrix(cocDiGraph, techniques):
    print('*** printing the adjacency matrix ***')
    # print(dod)
    nodelists = list(cocDiGraph.nodes())

    Index = nodelists
    Columns = nodelists

    IndexWithTTPsObject = []
    for idx in Index:
        te = next((x for x in techniques if x.id == idx), None)
        IndexWithTTPsObject.append(te)

    IndexWithTTPsObject.sort(key = lambda v : v.id)
    IndexWithTTPsObject.sort(key = lambda v : v.tactics[0].sequence)
    Index = [f'{x.id}' for x in IndexWithTTPsObject]
    Columns = [f'{x.id}' for x in IndexWithTTPsObject]

    # for item in IndexWithTTPsObject:
    #     print(f'{item.id}: {item.name} | {item.tactics[0].id}: {item.tactics[0].name}')

    df = pd.DataFrame(index=Index, columns=Columns)

    for ix in Index:
        for cl in Columns:
            if cocDiGraph.has_edge(ix, cl):
                df.at[ix, cl] = '*'
            else:
                df.at[ix, cl] = ' '

    print(tb.tabulate(df, headers='keys', showindex=True, tablefmt='psql'))


# get technique centrality measures (table IX)
def getTechniqueCentrality(cocDiGraph : nx.DiGraph, techniques : List['domain.Technique'], tactics : List['domain.Tactic']):
    
    dcofNodes = nx.degree_centrality(cocDiGraph)
    bcOfNodes = nx.betweenness_centrality(cocDiGraph, normalized = False)
    
    # pcInOfNodes = nx.katz_centrality(cocDiGraph, normalized = True)
    # pcOutOfNodes = nx.katz_centrality(cocDiGraph.reverse(), normalized = True)
    
    # ccInOfNodes = nx.closeness_centrality(cocDiGraph, wf_improved=True)
    # ccOutOfNodes = nx.closeness_centrality(cocDiGraph.reverse(), wf_improved=True)
    
    techniqueIds = []
    techniqueNames = []
    dcinvalues = []
    dcoutvalues = []
    # ccinvalues = []
    # ccoutvalues = []
    bcvalues = []
    # pcinvalues = []
    # pcoutvalues = []
    tacticsList = []
    
    techniuqesInGraph = [x for x in techniques if x.id in list(cocDiGraph.nodes())]
    
    for te in techniuqesInGraph:
        techniqueIds.append(te.id)
        techniqueNames.append(te.name)
        
        dcinvalues.append((cocDiGraph.in_degree[te.id]))
        dcoutvalues.append((cocDiGraph.out_degree[te.id]))
        # ccinvalues.append(ccInOfNodes[te.id])
        # ccoutvalues.append(ccOutOfNodes[te.id])
        bcvalues.append(bcOfNodes[te.id])
        # pcinvalues.append(pcInOfNodes[te.id])
        # pcoutvalues.append(pcOutOfNodes[te.id])
        
        # ta = next((x for x in tactics if x in te.tactics), None)
        # tacticsList.append(ta.id)
        
        ta = cocDiGraph.nodes[te.id]['tactic']
        tacticsList.append([x.name for x in tactics if x.id == ta][0])
        
    x=0
    data = {
        'id' : techniqueIds, 
        'Technique': techniqueNames, 
        'Tactic': tacticsList, 
        'IDC': dcinvalues,
        'ODC': dcoutvalues,  
        # 'bc': utils.normalizeList(bcvalues), 
        # 'cci': utils.normalizeList(ccinvalues),
        # 'cco': utils.normalizeList(ccoutvalues),
        # 'pci': utils.normalizeList(pcinvalues),
        # 'pco': utils.normalizeList(pcoutvalues)
        'BC': bcvalues, 
        # 'ICC': ccinvalues,
        # 'OCC': ccoutvalues,
        # 'IKC': pcinvalues,
        # 'OKC': pcoutvalues
    }
    
    df = pd.DataFrame(data)
    df = df.round(2)
    
    tacticnames = df['Tactic'].tolist()
    print(f"dci: {min(dcinvalues)} {statistics.mean(dcinvalues)} {statistics.quantiles(dcinvalues, n=4)}")
    print(f"dco: {min(dcoutvalues)} {statistics.mean(dcoutvalues)} {statistics.quantiles(dcoutvalues, n=4)}")
    print(f"bc: {min(bcvalues)} {statistics.mean(bcvalues)} {statistics.quantiles(bcvalues, n=4)}")
    # print(f"cci: {min(ccinvalues)} {statistics.mean(ccinvalues)} {statistics.quantiles(ccinvalues, n=4)}")
    # print(f"cco: {min(ccoutvalues)} {statistics.mean(ccoutvalues)} {statistics.quantiles(ccoutvalues, n=4)}")
    # print(f"kci: {min(pcinvalues)} {statistics.mean(pcinvalues)} {statistics.quantiles(pcinvalues, n=4)}")
    # print(f"kco: {min(pcoutvalues)} {statistics.mean(pcoutvalues)} {statistics.quantiles(pcoutvalues, n=4)}")
    
    print(f'*** centrality of techniques ***')
    print(tb.tabulate(df.sort_values(by=['IDC'], ascending=False).head(60), headers='keys', showindex=False, tablefmt='psql'))
    print(f'*** centrality of techniques stat ***')
    # print(tb.tabulate(df.describe(), headers='keys', tablefmt='psql'))
    
    # cclist = df['pc'].tolist()
    # print(np.percentile(cclist, 75))
    
    # print(tb.tabulate(df.corr().round(2),headers='keys', showindex=True, tablefmt='latex'))
    
    # dfm = pd.melt(df, id_vars=['id'], value_vars=['dc', 'cc', 'bc', 'pc'])
    # sns.violinplot(data=dfm, x='variable', y='value', inner='quartile',)
    # plt.show()
    
    
    dfq = df.query('Tactic == "TA0011"')
    # print(df.describe())
    
    ranges = []
    nodeCounts = []
    hueList = []
    
    bcvalues = utils.normalizeList(bcvalues)
    # ccinvalues = utils.normalizeList(ccinvalues)
    # pcinvalues = utils.normalizeList(pcinvalues)
    # ccoutvalues = utils.normalizeList(ccoutvalues)
    # pcoutvalues = utils.normalizeList(pcoutvalues)
    
    for i in range(0, 100, 20):
        nodeCount = len([x for x in bcvalues if x >= i/100 and x < (i+20)/100])
        ranges.append(f'{i/100}-{(i+20)/100}')
        hueList.append('BC')
        nodeCounts.append(nodeCount)
    return


findTopTenTechniques(techniques, cocTTPs)
findTopTactics(techniques, tactics, cocTTPs)
getTopTenCombinations(cocTTPs)
getTopTenSimpleRules(cocTTPs)
getTopTenCompoundRules(cocTTPs)
getAdjacencyMatrix(cocDiGraph, techniques)
getTechniqueCentrality(cocDiGraph, techniques, tactics)