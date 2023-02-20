import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import argparse
import os

#np.random.seed(4811) # uncomment to make reproducible

class CausalModel:
    """ A class representing a causal structural model."""
    
    def __init__(self):
        """ Write a proper initializer later."""
        self.nodes = []
        self.noises = [] # Gaussian for now
        self.edges = []
        self.coeffs = [] # Linear for now
        self.edgevars = []
        
        self.name = ""
        
        self.noisedict = dict(zip(self.nodes,self.noises))
        self.coeffdict = dict(zip(self.edges,self.coeffs))
        
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edges)
        #self.nodes = self.graph.nodes()
        self.numnodes = self.graph.number_of_nodes()
        self.numedges = self.graph.number_of_edges()
        
        self.topnodes = [n for n in nx.topological_sort(self.graph)]
        
    def refresh(self):
        self.noisedict = dict(zip(self.nodes,self.noises))
        self.coeffdict = dict(zip(self.edges,self.coeffs))
        
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edges)
        #self.nodes = self.graph.nodes()
        self.numnodes = self.graph.number_of_nodes()
        self.numedges = self.graph.number_of_edges()
        
        self.topnodes = [n for n in nx.topological_sort(self.graph)]
        
    def is_directed(self):
        return nx.is_directed(self.graph)
        
    def is_directed_acyclic_graph(self):
        return nx.is_directed_acyclic_graph(self.graph)
        
    def plot(self,tarrows=True,fname=""):
        plt.tight_layout()
        nx.draw_networkx(self.graph, arrows=tarrows)
        
        if fname:
            plt.savefig(fname+".png", format="PNG")
        plt.show()
        plt.clf()
        
    def get_ancestors(self,node):
        return nx.ancestors(self.graph,node)
    
    def get_parents(self,node):
        parent_iter = self.graph.predecessors(node)
        return [parent for parent in parent_iter]
        
    def get_children(self,node):
        child_iter = self.graph.successors(node)
        return [child for child in child_iter]
        
    def get_descendants(self,node):
        return nx.descendants(self.graph,node)
        
    def get_adjacency_matrix(self):
        return nx.to_numpy_matrix(self.graph)
        
    def get_node_distribution(self,node,condition_nodes={}):
        """ Returns Gaussian parameters (mean, variance) for node, flowing recursively over      
        all ancestors of node. Conditioned nodes are specified as dictionary condition_nodes."""
        if (node in condition_nodes):
            raise ValueError("Node of interest cannot also be conditioned on.")
    
        parents = self.get_parents(node)
        (mean,var) = self.noisedict[node]
        
        if not parents:         # If a node has no parents, it is entirely determined by noise
            return (mean,var)
            
        else:
            for parent in parents:
                thiscoeff = self.coeffdict[(parent,node)]
                if (parent in condition_nodes):
                    mean += thiscoeff * condition_nodes[parent]
                else:    
                    (thismean, thisvar) = self.get_node_distribution(parent)
                    mean += thiscoeff * thismean
                    var += thiscoeff**2 * thisvar
            return (mean,var)
            
    def sample(self,n=1):
        samples = []
        for i in range(n):
            sample = {}
            for node in self.topnodes:
                (thismean, thisvar) = self.noisedict[node]
                val = np.random.normal(thismean, thisvar)
                
                parents = self.get_parents(node)
                for parent in parents:
                    thiscoeff = self.coeffdict[(parent,node)]
                    val += thiscoeff * sample[parent]
                    
                sample[node] = val
            samples.append(sample)
        return samples

class CausalModelPoly:
    """ A class representing a causal structural model with polynomial couplings."""
    
    def __init__(self):
        """ Write a proper initializer later."""
        self.nodes = []
        self.noises = [] # Gaussian for now
        self.edges = []
        self.coeffs = [] # Each "coeff" will now be a list of polynomial coeffs, increasing order
        self.edgevars = []
        
        self.name = ""
        
        self.noisedict = dict(zip(self.nodes,self.noises))
        self.coeffdict = dict(zip(self.edges,self.coeffs))
        
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edges)
        #self.nodes = self.graph.nodes()
        self.numnodes = self.graph.number_of_nodes()
        self.numedges = self.graph.number_of_edges()
        
        self.topnodes = [n for n in nx.topological_sort(self.graph)]
        
    def refresh(self):
        self.noisedict = dict(zip(self.nodes,self.noises))
        self.coeffdict = dict(zip(self.edges,self.coeffs))
        
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edges)
        #self.nodes = self.graph.nodes()
        self.numnodes = self.graph.number_of_nodes()
        self.numedges = self.graph.number_of_edges()
        
        self.topnodes = [n for n in nx.topological_sort(self.graph)]
        
    def is_directed(self):
        return nx.is_directed(self.graph)
        
    def is_directed_acyclic_graph(self):
        return nx.is_directed_acyclic_graph(self.graph)
        
    def plot(self,tarrows=True,fname=""):
        plt.tight_layout()
        nx.draw_networkx(self.graph, arrows=tarrows)
        
        if fname:
            plt.savefig(fname+".png", format="PNG")
        plt.show()
        plt.clf()
        
    def get_ancestors(self,node):
        return nx.ancestors(self.graph,node)
    
    def get_parents(self,node):
        parent_iter = self.graph.predecessors(node)
        return [parent for parent in parent_iter]
        
    def get_children(self,node):
        child_iter = self.graph.successors(node)
        return [child for child in child_iter]
        
    def get_descendants(self,node):
        return nx.descendants(self.graph,node)
        
    def get_adjacency_matrix(self):
        return nx.to_numpy_matrix(self.graph)
        
    def get_node_distribution(self,node,condition_nodes={}):
        """ Returns Gaussian parameters (mean, variance) for node, flowing recursively over      
        all ancestors of node. Conditioned nodes are specified as dictionary condition_nodes."""
        if (node in condition_nodes):
            raise ValueError("Node of interest cannot also be conditioned on.")
    
        parents = self.get_parents(node)
        (mean,var) = self.noisedict[node]
        
        if not parents:         # If a node has no parents, it is entirely determined by noise
            return (mean,var)
            
        else:
            for parent in parents:
                thiscoeffs = self.coeffdict[(parent,node)]
                if (parent in condition_nodes):
                    for n,thiscoeff in enumerate(thiscoeffs):
                        mean += thiscoeff * (condition_nodes[parent])**n
                else:    
                    (thismean, thisvar) = self.get_node_distribution(parent)
                    for n,thiscoeff in enumerate(thiscoeffs):
                        mean += thiscoeff * thismean**n
                    #var += thiscoeff**2 * thisvar # let's avoid calculus for now
                return (mean,var)
            
    def sample(self,n=1):
        samples = []
        for i in range(n):
            sample = {}
            for node in self.topnodes:
                (thismean, thisvar) = self.noisedict[node]
                val = np.random.normal(thismean, thisvar)
                
                parents = self.get_parents(node)
                for parent in parents:
                    thiscoeffs = self.coeffdict[(parent,node)]
                    for n,thiscoeff in enumerate(thiscoeffs):
                        val += thiscoeff * (sample[parent])**n
                    
                sample[node] = val
            samples.append(sample)
        return samples
               
            

def generate_AtheyModel(confounding=0):          
                
        
    model = CausalModel()
    model.nodes = ["W","S","Y","X","U"]
    model.noises = [(0,1),(0,1),(0,1),(0,1),(0,1)]
    model.edges = [("W","S"),("S","Y"),("X","W"),("X","S"),("X","Y"),("U","W"),("U","Y"),("U","S")]
    model.coeffs = [1,1,1,1,1,1,1,confounding]
    model.edgevars = []
    model.name = "Athey Model"
    model.refresh()

    if not model.is_directed():
        raise ValueError("Graph input is undirected.")
    if not model.is_directed_acyclic_graph():
        raise ValueError("Graph input contains one or more cycles.")
        
    return model

def generate_ConfounderModel(coeffs,noise=1,shift=0):          
                
        
    model = CausalModel()
    model.nodes = ["W","X","M","Y"]
    model.noises = [(shift,noise),(0,noise),(0,noise),(0,noise)]
    model.edges = [("W","X"),("X","M"),("M","Y"),("W","Y"),("W","M")]
    model.coeffs = coeffs
    model.edgevars = ["d","c","a","b","e"]
    model.name = "Confounded-Mediator (CM) Model"
    model.refresh()

    if not model.is_directed():
        raise ValueError("Graph input is undirected.")
    if not model.is_directed_acyclic_graph():
        raise ValueError("Graph input contains one or more cycles.")
        
    return model   
    
def generate_InstrumentModel(coeffs,noise=1,shift=0):          
                
        
    model = CausalModel()
    model.nodes = ["W","V","X","M","Y"]
    model.noises = [(shift,noise),(0,noise),(0,noise),(0,noise),(0,noise)]
    model.edges = [("W","X"),("X","M"),("M","Y"),("W","Y"),("V","X"),("W","M")]
    model.coeffs = coeffs
    model.edgevars = ["d","c","a","b","g","e"]
    model.name = "Leading-Instrument CM Model"
    model.refresh()
    
    print(model.nodes)
    print(model.edges)
    print(model.graph.nodes)
    print(model.topnodes)

    if not model.is_directed():
        raise ValueError("Graph input is undirected.")
    if not model.is_directed_acyclic_graph():
        raise ValueError("Graph input contains one or more cycles.")
        
    #model.plot(tarrows=True,fname="instrumentgraph")
        
    return model
    
def generate_ConfounderModelPoly(coeffs,noise=1,xnoise=1,shift=0):          
    # Coeffs is now a list of list of increasing-ordered polynomial coefficients
        
    model = CausalModelPoly()
    model.nodes = ["W","X","M","Y"]
    model.noises = [(shift,noise),(0,noise),(0,noise),(0,noise)]
    model.edges = [("W","X"),("X","M"),("M","Y"),("W","Y"),("W","M")]
    model.coeffs = coeffs
    model.edgevars = ["d","c","a","b","e"]
    model.name = "Confounded-Mediator (CM) Model with Polynomial Couplings"
    model.refresh()

    if not model.is_directed():
        raise ValueError("Graph input is undirected.")
    if not model.is_directed_acyclic_graph():
        raise ValueError("Graph input contains one or more cycles.")
        
    return model 

def generate_InstrumentModelPoly(coeffs,noise=1,xnoise=1,shift=0):          
    # Coeffs is now a list of list of increasing-ordered polynomial coefficients            
        
    model = CausalModelPoly()
    model.nodes = ["W","V","X","M","Y"]
    model.noises = [(shift,noise),(0,0),(0,xnoise),(0,noise),(0,noise)]
    model.edges = [("W","X"),("X","M"),("M","Y"),("W","Y"),("V","X"),("W","M")]
    model.coeffs = coeffs
    model.edgevars = ["d","c","a","b","g","e"]
    model.name = "Leading-Instrument CM Model with Polynomial Couplings"
    model.refresh()
    
    print(model.nodes)
    print(model.edges)
    print(model.graph.nodes)
    print(model.topnodes)

    if not model.is_directed():
        raise ValueError("Graph input is undirected.")
    if not model.is_directed_acyclic_graph():
        raise ValueError("Graph input contains one or more cycles.")
        
    #model.plot(tarrows=True,fname="instrumentgraph")
        
    return model
    
