---
jupyter:
  jupytext:
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

# Quiz 4

BEFORE YOU START THIS QUIZ:

1. Click on "Copy to Drive" to make a copy of the quiz

2. Click on "Share",
    
3. Click on "Change" and select "Anyone with this link can edit"
    
4. Click "Copy link" and

5. Paste the link into [this Canvas assignment](https://canvas.olin.edu/courses/315/assignments/5071).

This quiz is based on Chapters 9 and 10 of [Think Complexity, 2nd edition](https://thinkcomplex.com).

Copyright 2021 Allen Downey, [MIT License](http://opensource.org/licenses/MIT)

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
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from utils import decorate
```

## Fireflies

In Chapter 1 of Think Complexity, I wrote about Strogatz's model of spontaneous synchronization in some species of fireflies and his proof that “sync was inevitable”. I said:

> Strogatz makes several simplifying assumptions, in particular that each firefly can see all the others. In my opinion, it is more interesting to explain how an entire valley of fireflies can synchronize despite the fact that they cannot all see each other.
>
>Explanations of these phenomena often use agent-based models, which explore ... the conditions that allow or prevent synchronization.

So let's get back to that.

Among many other awesome projects, [Nicky Case](https://ncase.me/faq/) has developed an online simulation of firefly synchronization, [which you can run here](https://ncase.me/fireflies/).


The goal of this activity is to replicate their model and explore its behavior as we vary some of its parameters.

Here are the steps:

1) Try out the simulation and read the text (in the left margin of the page) that explains the model.

2) Think about what aspects of the model you will replicate and what you will leave out.

3) Think about a parameter you might vary to see what effect it has.

4) Implement the model and see if it exhibits spontaneous synchronization.

5) Identify a parameter that can control whether synchronization occurs or not.


You should spend a minimum of 1.5 on this activity and a maximum of 3 hours. It might not be possible to complete all five steps in this time, but that's ok.
The goals of this activity are to practice the initial stages of a project and to develop rapid prototyping skills.

Some design decisions you should consider:

* What is the nature of the space the fireflies live in? It could be a grid, a network, or continuous space in 1, 2, or 3 dimensions.

* Do fireflies move in space, and how?

* Which fireflies can "see" each other, and how do they interact?

* How will you display the results, and how will you determine whether syncronization has occurred. (Animated displays are cool, but they might not be necessary and they can be a time sink.)


You are welcome to use external resources for this activity, but you should give credit to any resources you make substantial use of.

You can discuss your plans with the instructor and other students. You can get help debugging, if needed. But the result should be substantially your own, independent work.

Assuming that, as a class, we explore a variety of models and implementations, we should have some interesting results to compare!

```python

```
