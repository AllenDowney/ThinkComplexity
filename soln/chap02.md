---
jupyter:
  jupytext:
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

The second edition of *Think Complexity* is available from [Bookshop.org](https://bookshop.org/a/98697/9781492040200) and [Amazon](https://amzn.to/3wPN0SJ) (those are affiliate links). If you are enjoying the free, online version, consider [buying me a coffee](https://buymeacoffee.com/allendowney).

# Graphs

The next three chapters are about systems made up of components and connections between components. For example, in a social network, the components are people and connections represent friendships, business relationships, etc. In an ecological food web, the components are species and the connections represent predator-prey relationships.

In this chapter, I introduce NetworkX, a Python package for building models of these systems. We start with the Erdős-Rényi model, which has interesting mathematical properties. In the next chapter we move on to models that are more useful for explaining real-world systems.


[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkComplexity/blob/v3/soln/chap02.ipynb).

```python
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
```

```python
from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)
    
download('https://github.com/AllenDowney/ThinkComplexity/raw/v3/nb/utils.py')
```

```python
from utils import decorate, savefig

# Set the random seed so the notebook 
# produces the same results every time.
np.random.seed(17)
```

```python
# make a directory for figures
!mkdir -p figs
```

```python
# node colors for drawing networks
colors = sns.color_palette('pastel', 5)
#sns.palplot(colors)
sns.set_palette(colors)
```

## What is a graph?

To most people a "graph" is a visual representation of data, like a bar chart or a plot of stock prices over time. That's not what this chapter is about.

In this chapter, a **graph** is a representation of a system that contains discrete, interconnected elements. The elements are represented by **nodes** — also called **vertices** — and the interconnections are represented by **edges**.

For example, you could represent a road map with a node for each city and an edge for each road between cities. Or you could represent a social network using a node for each person, with an edge between two people if they are friends.

In some graphs, edges have attributes like length, cost, or weight. For example, in a road map, the length of an edge might represent distance between cities or travel time. In a social network there might be different kinds of edges to represent different kinds of relationships: friends, business associates, etc.


Edges may be **directed** or **undirected**, depending on whether the relationships they represent are asymmetric or symmetric. In a road map, you might represent a one-way street with a directed edge and a two-way street with an undirected edge. In some social networks, like Facebook, friendship is symmetric: if $A$ is friends with $B$ then $B$ is friends with $A$. But on Twitter, for example, the "follows" relationship is not symmetric; if $A$ follows $B$, that doesn't imply that $B$ follows $A$. So you might use undirected edges to represent a Facebook network and directed edges for Twitter.

Graphs have interesting mathematical properties, and there is a branch of mathematics called **graph theory** that studies them.

Graphs are also useful, because there are many real world problems that can be solved using **graph algorithms**. For example, Dijkstra's shortest path algorithm is an efficient way to find the shortest path from a node to all other nodes in a graph. A **path** is a sequence of nodes with an edge between each consecutive pair.

Graphs are usually drawn with squares or circles for nodes and lines for edges. In the next section, we'll draw a directed graph that represents a social network, and an undirected graph that represents driving times between cities.

## NetworkX

To represent graphs, we'll use a package called NetworkX, which is the most commonly used network library in Python. You can read more about it at <https://thinkcomplex.com/netx>, but I'll explain it as we go along.

We can create a directed graph by importing NetworkX (usually imported as `nx`) and instantiating `nx.DiGraph`:

```python
G = nx.DiGraph()
```

At this point, `G` is a `DiGraph` object that contains no nodes and no edges. We can add nodes using the `add_node` method:

```python
G.add_node('Alice')
G.add_node('Bob')
G.add_node('Cate')
```

Now we can use the `nodes` method to get a list of nodes:

```python
list(G.nodes())
```

The `nodes` method returns a `NodeView`, which can be used in a for loop or, as in this example, used to make a list.

Adding edges works pretty much the same way:

```python
G.add_edge('Alice', 'Bob')
G.add_edge('Alice', 'Cate')
G.add_edge('Bob', 'Alice')
G.add_edge('Bob', 'Cate')
```

And we can use `edges` to get the list of edges:

```python
list(G.edges())
```

NetworkX provides several functions for drawing graphs; `draw_circular` arranges the nodes in a circle and connects them with edges:

```python
nx.draw_circular(G,
                 node_color='C0',
                 node_size=2000, 
                 with_labels=True)
plt.axis('equal')
savefig('figs/chap02-1')
```

This directed graph might represent three people who follow each other on Twitter. The arrow indicates the direction of the relationship. In this example, Alice and Bob follow each other, both follow Cate, and Cate follows no one.

The option `with_labels` causes the nodes to be labeled; in the next example we'll see how to label the edges.

**Exercise:**  Add another node and a few more edges and draw the graph again.

```python
# Solution

G.add_edge('Dave', 'Alice')
G.add_edge('Dave', 'Cate')
nx.draw_circular(G,
                 node_color='C0',
                 node_size=2000, 
                 with_labels=True)
```

The second example is an undirected graph that represents four cities in the northeast United States and the driving times between them. I start with a dictionary that maps from each city name to its approximate longitude and latitude:

```python
positions = dict(Albany=(-74, 43),
                 Boston=(-71, 42),
                 NYC=(-74, 41),
                 Philly=(-75, 40))

positions['Albany']
```

Since this is an undirected graph, I instantiate `nx.Graph`. Then I can use `add_nodes_from` to iterate the keys of `positions` and add them as nodes:

```python
G = nx.Graph()
G.add_nodes_from(positions)
G.nodes()
```

Next I'll make a dictionary that maps from each edge to the corresponding driving time:

```python
drive_times = {('Albany', 'Boston'): 3,
               ('Albany', 'NYC'): 4,
               ('Boston', 'NYC'): 4,
               ('NYC', 'Philly'): 2}
```

Now I can use `add_edges_from`, which iterates the keys of `drive_times` and adds them as edges:

```python
G.add_edges_from(drive_times)
G.edges()
```

Instead of `draw_circular`, which arranges the nodes in a circle, I'll use `draw`, which takes the position dictionary as the second parameter. `draw` uses `positions` to determine the locations of the nodes.

To add the edge labels, we use `draw_networkx_edge_labels`. The `edge_labels` parameter expects a dictionary that maps from each pair of nodes to a label; in this case, the labels are driving times between cities.

```python
nx.draw(G, positions, 
        node_color='C1', 
        node_shape='s', 
        node_size=2500, 
        with_labels=True)

nx.draw_networkx_edge_labels(G, positions, 
                             edge_labels=drive_times)

plt.axis('equal')
savefig('figs/chap02-2')
```

The labels on the edges indicate driving time in hours. In this example the placement of the nodes corresponds roughly to the geography of the cities, but in general the layout of a graph is arbitrary.

In both of these examples, the nodes are strings, but in general they can be any hashable type.

**Exercise:**  Add another city and at least one edge.

```python
# Solution

positions['Scranton'] = (-76, 42)
G.add_node('Scranton')

drive_times.update({('Scranton', 'Albany'): 3,
                    ('Scranton', 'NYC'): 2,
                    ('Scranton', 'Philly'): 2})

G.add_edges_from(drive_times)

nx.draw(G, positions, 
        node_color='C1', 
        node_shape='s', 
        node_size=2500, 
        with_labels=True)

nx.draw_networkx_edge_labels(G, positions, 
                             edge_labels=drive_times)

plt.axis('equal')
```

## Random graphs

A random graph is just what it sounds like: a graph with nodes and edges generated at random. Of course, there are many random processes that can generate graphs, so there are many kinds of random graphs.

One of the more interesting kinds is the Erdős-Rényi model, studied by Paul Erdős and Alfréd Rényi in the 1960s.

An Erdős-Rényi graph (ER graph) is characterized by two parameters: $n$ is the number of nodes and $p$ is the probability that there is an edge between any two nodes. See <https://thinkcomplex.com/er>.

Erdős and Rényi studied the properties of these random graphs; one of their surprising results is the existence of abrupt changes in the properties of random graphs as random edges are added.


One of the properties that displays this kind of transition is connectivity. An undirected graph is **connected** if there is a path from every node to every other node.

In an ER graph, the probability that the graph is connected is very low when $p$ is small and nearly 1 when $p$ is large. Between these two regimes, there is a rapid transition at a particular value of $p$, denoted $p^*$.

Erdős and Rényi showed that this critical value is $p^* = (\ln n) / n$, where $n$ is the number of nodes. A random graph, $G(n, p)$, is unlikely to be connected if $p < p^*$ and very likely to be connected if $p > p^*$.

To test this claim, we'll develop algorithms to generate random graphs and check whether they are connected.

## Generating graphs

I'll start by generating a **complete** graph, which is a graph where every node is connected to every other.

Here's a generator function that takes a list of nodes and enumerates all distinct pairs. If you are not familiar with generator functions, you can read about them at <https://thinkcomplex.com/gen>.

```python
def all_pairs(nodes):
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j:
                yield u, v
```

We can use `all_pairs` to construct a complete graph:

```python
def make_complete_graph(n):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_pairs(nodes))
    return G
```

`make_complete_graph` takes the number of nodes, `n`, and returns a new `Graph` with `n` nodes and edges between all pairs of nodes.

The following code makes a complete graph with 10 nodes:

```python
complete = make_complete_graph(10)
complete.number_of_nodes()
```

And here's what it looks like.

```python
nx.draw_circular(complete, 
                 node_color='C2', 
                 node_size=1000, 
                 with_labels=True)
savefig('figs/chap02-3')
```

Soon we will modify this code to generate ER graphs, but first we'll develop functions to check whether a graph is connected.

**Exercise:** Make and draw a complete directed graph with 5 nodes.

```python
# Solution

def all_directed_pairs(nodes):
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i != j:
                yield u, v
                
def make_complete_digraph(n):
    G = nx.DiGraph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_directed_pairs(nodes))
    return G

complete_digraph = make_complete_digraph(5)

nx.draw_circular(complete_digraph, 
                 node_color='C2', 
                 node_size=1000, 
                 with_labels=True)
```

## Connected graphs

A graph is **connected** if there is a path from every node to every other node (see <https://thinkcomplex.com/conn>).

For many applications involving graphs, it is useful to check whether a graph is connected. Fortunately, there is a simple algorithm that does it.

You can start at any node and check whether you can reach all other nodes. If you can reach a node, $v$, you can reach any of the **neighbors** of $v$, which are the nodes connected to $v$ by an edge.

The `Graph` class provides a method called `neighbors` that returns an iterator over the neighbors of a given node. For example, in the complete graph we generated in the previous section:

```python
list(complete.neighbors(0))
```

Suppose we start at node $s$. We can mark $s$ as "seen" and mark its neighbors. Then we mark the neighbors' neighbors, and their neighbors, and so on, until we can't reach any more nodes. If all nodes are seen, the graph is connected.

Here's what that looks like in Python:

```python
def reachable_nodes(G, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen
```

`reachable_nodes` takes a `Graph` and a starting node, `start`, and returns the set of nodes that can be reached from `start`.

Initially the set, `seen`, is empty, and we create a list called `stack` that keeps track of nodes we have discovered but not yet processed. Initially the stack contains a single node, `start`.

Now, each time through the loop, we:

1.  Remove one node from the stack.

2.  If the node is already in `seen`, we go back to Step 1.

3.  Otherwise, we add the node to `seen` and add its neighbors to the stack.

When the stack is empty, we can't reach any more nodes, so we break out of the loop and return `seen`.

As an example, we can find all nodes in the complete graph that are reachable from node 0:

```python
reachable_nodes(complete, 0)
```

Initially, the stack contains node 0 and `seen` is empty. The first time through the loop, node 0 is added to `seen` and all the other nodes are added to the stack (since they are all neighbors of node 0).

The next time through the loop, `pop` returns the last element in the stack, which is node 9. So node 9 gets added to `seen` and its neighbors get added to the stack.

Notice that the same node can appear more than once in the stack; in fact, a node with $k$ neighbors will be added to the stack $k$ times. Later, we will look for ways to make this algorithm more efficient.

We can use `reachable_nodes` to write `is_connected`:

```python
def is_connected(G):
    start = next(iter(G))
    reachable = reachable_nodes(G, start)
    return len(reachable) == len(G)
```

`is_connected` chooses a starting node by making a node iterator and choosing the first element. Then it uses `reachable_nodes` to get the set of nodes that can be reached from `start`. If the size of this set is the same as the size of the graph, that means we can reach all nodes, which means the graph is connected.

A complete graph is, not surprisingly, connected:

```python
is_connected(complete)
```

In the next section we will generate ER graphs and check whether they are connected.

**Exercise:** What do you think it means for a directed graph to be connected?  Write a function that checks whether a directed graph is connected.

```python
# Solution

"""According to [Wolfram MathWorld](http://mathworld.wolfram.com/ConnectedDigraph.html)

"There are two distinct notions of connectivity in a directed graph. A directed graph is 
weakly connected if there is an undirected path between any pair of vertices, 
and strongly connected if there is a directed path between every pair of vertices"

We'll check for strong connectedness.

"""

def directed_reachable_nodes(G, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.successors(node))
    return seen

def digraph_is_connected(G):
    for start in G:
        reachable = directed_reachable_nodes(G, start)
        if len(reachable) < len(G):
            return False
    return True
        
complete_digraph = make_complete_digraph(5)
digraph_is_connected(complete_digraph)
```

## Generating ER graphs

The ER graph $G(n, p)$ contains $n$ nodes, and each pair of nodes is connected by an edge with probability $p$. Generating an ER graph is similar to generating a complete graph.

The following generator function enumerates all possible edges and chooses which ones should be added to the graph:

```python
def random_pairs(nodes, p):
    for edge in all_pairs(nodes):
        if flip(p):
            yield edge
```

`random_pairs` uses `flip`:

```python
def flip(p):
    return np.random.random() < p
```

This is the first example we've seen that uses NumPy. Following convention, I import `numpy` as `np`. NumPy provides a module named `random`, which provides a method named `random`, which returns a number between 0 and 1, uniformly distributed.

So `flip` returns `True` with the given probability, `p`, and `False` with the complementary probability, `1-p`.

Finally, `make_random_graph` generates and returns the ER graph $G(n, p)$:

```python
def make_random_graph(n, p):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(random_pairs(nodes, p))
    return G
```

`make_random_graph` is almost identical to `make_complete_graph`; the only difference is that it uses `random_pairs` instead of `all_pairs`.

Here's an example with `n=10` and `p=0.3`:

```python
np.random.seed(10)

random_graph = make_random_graph(10, 0.3)
len(random_graph.edges())
```

And here's what it looks like:

```python
nx.draw_circular(random_graph, 
                 node_color='C3', 
                 node_size=1000, 
                 with_labels=True)
savefig('figs/chap02-4')
```

This graph turns out to be connected: starting from node 0, we can reach all nodes.

```python
reachable_nodes(random_graph, 0)
```

In fact, most ER graphs with $n=10$ and $p=0.3$ are connected. But if we generate a random graph with a low value of `p`, it's not:

```python
random_graph = make_random_graph(10, 0.1)
len(random_graph.edges())
```

```python
is_connected(random_graph)
```

In the next section, we'll see how many are connected.

## Probability of connectivity

For given values of $n$ and $p$, we would like to know the probability that $G(n, p)$ is connected. We can estimate it by generating a large number of random graphs and counting how many are connected. This function takes `n` and `p`, generates `iters` graphs, and returns the fraction of them that are connected.

```python
# version with a for loop

def prob_connected(n, p, iters=100):
    count = 0
    for i in range(iters):
        random_graph = make_random_graph(n, p)
        if is_connected(random_graph):
            count += 1
    return count/iters
```

Here's a more concise version:

```python
# version with a list comprehension

def prob_connected(n, p, iters=100):
    tf = [is_connected(make_random_graph(n, p))
          for i in range(iters)]
    return np.mean(tf)
```

The parameters `n` and `p` are passed along to `make_random_graph`; `iters` is the number of random graphs we generate.

This function uses a list comprehension; if you are not familiar with this feature, you can read about it at <https://thinkcomplex.com/comp>.

The result, `tf`, is a list of boolean values: `True` for each graph that's connected and `False` for each one that's not.

`np.mean` is a NumPy function that computes the mean of this list, treating `True` as 1 and `False` as 0. The result is the fraction of random graphs that are connected.

```python
np.random.seed(17)

n = 10
prob_connected(n, 0.23, iters=10000)
```

I chose $0.23$ because it is close to the critical value where the probability of connectivity goes from near 0 to near 1. According to Erdős and Rényi, $p^* = \ln n / n = 0.23$.

```python
pstar = np.log(n) / n
pstar
```

We can get a clearer view of the transition by estimating the probability of connectivity for a range of values of $p$:

```python
ps = np.logspace(-1.3, 0, 11)
ps
```

The NumPy function `logspace` returns an array of 11 values from $10^{-1.3}$ to $10^0 = 1$, equally spaced on a logarithmic scale.

For each value of `p` in the array, we compute the probability that a graph with parameter `p` is connected and store the results in `ys`. I'll estimate the probabilities with `iters=1000`.

```python
ys = [prob_connected(n, p, 1000) for p in ps]

for p, y in zip(ps, ys):
    print(p, y)
```

The following figure shows the results, with a vertical line at the computed critical value, $p^* = 0.23$.

```python
plt.axvline(pstar, color='gray')
plt.plot(ps, ys, color='green')
decorate(xlabel='Prob of edge (p)',
                 ylabel='Prob connected',
                 xscale='log')

savefig('figs/chap02-5')
```

As expected, the transition from 0 to 1 occurs near the critical value.

We can run the same analysis for a few more values of `n`. The following figure shows the results, with vertical lines at the critical values.

```python
ns = [300, 100, 30]
ps = np.logspace(-2.5, 0, 11)

sns.set_palette('Blues_r', 4)
for n in ns:
    print(n)
    pstar = np.log(n) / n
    plt.axvline(pstar, color='gray', alpha=0.3)

    ys = [prob_connected(n, p) for p in ps]
    plt.plot(ps, ys, label='n=%d' % n)

decorate(xlabel='Prob of edge (p)',
         ylabel='Prob connected',
         xscale='log', 
         xlim=[ps[0], ps[-1]],
         loc='upper left')

savefig('figs/chap02-6')
```

As $n$ increases, the critical value gets smaller and the transition gets more abrupt.

These experimental results are consistent with the analytic results Erdős and Rényi presented in their papers.

## Analysis of graph algorithms

Earlier in this chapter I presented an algorithm for checking whether a graph is connected; in the next few chapters, we will see other graph algorithms. Along the way, we will analyze the performance of those algorithms, figuring out how their run times grow as the size of the graphs increases.

If you are not already familiar with analysis of algorithms, you might want to read Appendix B of *Think Python, 2nd Edition*, at <https://thinkcomplex.com/tp2>.

The order of growth for graph algorithms is usually expressed as a function of $n$, the number of vertices (nodes), and $m$, the number of edges.

As an example, let's analyze `reachable_nodes` from the "Connected graphs" section:

```
def reachable_nodes(G, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen
```

Each time through the loop, we pop a node off the stack; by default, `pop` removes and returns the last element of a list, which is a constant time operation.

Next we check whether the node is in `seen`, which is a set, so checking membership is constant time.

If the node is not already in `seen`, we add it, which is constant time, and then add the neighbors to the stack, which is linear in the number of neighbors.


To express the run time in terms of $n$ and $m$, we can add up the total number of times each node is added to `seen` and `stack`.

Each node is only added to `seen` once, so the total number of additions is $n$.

But nodes might be added to `stack` many times, depending on how many neighbors they have. If a node has $k$ neighbors, it is added to `stack` $k$ times. Of course, if it has $k$ neighbors, that means it is connected to $k$ edges.

So the total number of additions to `stack` is the total number of edges, $m$, doubled because we consider every edge twice.

Therefore, the order of growth for this function is $O(n+ m)$, which is a convenient way to say that the run time grows in proportion to either $n$ or $m$, whichever is bigger.

If we know the relationship between $n$ and $m$, we can simplify this expression. For example, in a complete graph the number of edges is $n(n-1)/2$, which is in $O(n^2)$. So for a complete graph, `reachable_nodes` is quadratic in $n$.

## Exercises

**Exercise:** In the "Analysis of graph algorithms" section we analyzed the performance of `reachable_nodes` and classified it in $O(n + m)$, where $n$ is the number of nodes and $m$ is the number of edges.  Continuing the analysis, what is the order of growth for `is_connected`?

    def is_connected(G):
        start = next(iter(G))
        reachable = reachable_nodes(G, start)
        return len(reachable) == len(G)

```python
# Solution

"""Creating an iterator and getting the first element are constant time operations.
Calling `reachable_nodes` is $O(n+m)$.  Getting the length of a set and a graph are 
constant time (but even if they were linear in $n$, it wouldn't matter).  So the 
total for `is_connected` is $O(n+m)$.""";
```

**Exercise:** In my implementation of `reachable_nodes`, you might be bothered by the apparent inefficiency of adding *all* neighbors to the stack without checking whether they are already in `seen`.  Write a version of this function that checks the neighbors before adding them to the stack.  Does this "optimization" change the order of growth?  Does it make the function faster?

```python
def reachable_nodes_precheck(G, start):
    # FILL THIS IN
    return []
```

```python
# Solution

"""Checking the nodes before putting them on the stack does not affect the order 
of growth; we have to perform the same number of checks either way.  But it might 
be a little faster because it avoids the overhead of adding and removing nodes 
from the stack over and over."""

def reachable_nodes_precheck(G, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            neighbors = set(G[node]) - seen
            stack.extend(neighbors)
    return seen

complete = make_complete_graph(100)
```

```python
%timeit len(reachable_nodes(complete, 0))
```

```python
%timeit len(reachable_nodes_precheck(complete, 0))
```

**Exercise:** There are actually two kinds of ER graphs.  The one we generated in this chapter, $G(n, p)$, is characterized by two parameters, the number of nodes and the probability of an edge between nodes.

An alternative definition, denoted $G(n, m)$, is also characterized by two parameters: the number of nodes, $n$, and the number of edges, $m$.  Under this definition, the number of edges is fixed, but their location is random.

Repeat the experiments we did in this chapter using this alternative definition.  Here are a few suggestions for how to proceed:

1. Write a function called `m_pairs` that takes a list of nodes and the number of edges, $m$, and returns a random selection of $m$ edges.  A simple way to do that is to generate a list of all possible edges and use `random.sample`.

2. Write a function called `make_m_graph` that takes $n$ and $m$ and returns a random graph with $n$ nodes and $m$ edges.

3. Make a version of `prob_connected` that uses `make_m_graph` instead of `make_random_graph`.

4. Compute the probability of connectivity for a range of values of $m$.

How do the results of this experiment compare to the results using the first type of ER graph?

```python
# Solution

colors = sns.color_palette('pastel', 5)
sns.set_palette(colors)

import random

def m_pairs(nodes, m):
    pairs = list(all_pairs(nodes))
    return random.sample(pairs, m)

def all_edges(nodes):
    for u in nodes:
        for v in nodes:
            if u < v:
                yield u, v

def m_pairs(nodes, m):
    n = len(nodes)
    num = m
    den = n * (n-1) / 2 
    for edge in all_edges(nodes):
        p = num / den
        if flip(p):
            num -= 1
            den -= 1
            yield edge
        else:
            den -= 1

def make_m_graph(n, m):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(m_pairs(nodes, m))
    return G

m_graph = make_m_graph(10, 47)

nx.draw_circular(m_graph, 
                 node_color='C4', 
                 node_size=1000, 
                 with_labels=True)
```

```python
m_graph.number_of_edges()
```

```python
# Solution

def prob_m_connected(n, m, iters=100):
    count = 0
    for i in range(iters):
        m_graph = make_m_graph(n, m)
        if is_connected(m_graph):
            count += 1
    return count/iters

n = 10
pstar = np.log(n) / n
ps = np.logspace(-1.3, 0, 11)
ms = [int(p * n * (n-1) / 2) for p in ps]
ys = [prob_m_connected(n, m, 100) for m in ms]

for p, m, y in zip(ps, ms, ys):
    print(p, m, y)
```

```python
# Solution

plt.axvline(pstar, color='gray')
plt.plot(ps, ys, color='green')
decorate(xlabel='Prob of edge (p)',
         ylabel='Prob connected',
         xscale='log', 
         xlim=[ps[0], ps[-1]])
```

```python

```

[Think Complexity, 2nd Edition](https://greenteapress.com/wp/think-complexity-2e/)

Copyright 2016 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
