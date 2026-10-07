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

# Complexity Science

Complexity science is relatively new; it became recognizable as a field, and was given a name, in the 1980s. But its newness is not because it applies the tools of science to a new subject, but because it uses different tools, allows different kinds of work, and ultimately changes what we mean by "science".

To demonstrate the difference, I'll start with an example of classical science: suppose someone asks you why planetary orbits are elliptical. You might invoke Newton's law of universal gravitation and use it to write a differential equation that describes planetary motion. Then you can solve the differential equation and show that the solution is an ellipse. QED!

Most people find this kind of explanation satisfying. It includes a mathematical derivation — so it has some of the rigor of a proof — and it explains a specific observation, elliptical orbits, by appealing to a general principle, gravitation.

Let me contrast that with a different kind of explanation. Suppose you move to a city like Detroit that is racially segregated, and you want to know why it's like that. If you do some research, you might find a paper by Thomas Schelling called "Dynamic Models of Segregation", which proposes a simple model of racial segregation:

Here is my description of the model, from Chapter 9:

> The Schelling model of the city is an array of cells where each cell represents a house. The houses are occupied by two kinds of "agents", labeled red and blue, in roughly equal numbers. About 10% of the houses are empty.
>
> At any point in time, an agent might be happy or unhappy, depending on the other agents in the neighborhood. In one version of the model, agents are happy if they have at least two neighbors like themselves, and unhappy if they have one or zero.
>
> The simulation proceeds by choosing an agent at random and checking to see whether it is happy. If so, nothing happens; if not, the agent chooses one of the unoccupied cells at random and moves.

If you start with a simulated city that is entirely unsegregated and run the model for a short time, clusters of similar agents appear. As time passes, the clusters grow and coalesce until there are a small number of large clusters and most agents live in homogeneous neighborhoods.

The degree of segregation in the model is surprising, and it suggests an explanation of segregation in real cities. Maybe Detroit is segregated because people prefer not to be greatly outnumbered and will move if the composition of their neighborhoods makes them unhappy.

Is this explanation satisfying in the same way as the explanation of planetary motion? Many people would say not, but why?

Most obviously, the Schelling model is highly abstract, which is to say not realistic. So you might be tempted to say that people are more complicated than planets. But that can't be right. After all, some planets have people on them, so they have to be more complicated than people.

Both systems are complicated, and both models are based on simplifications. For example, in the model of planetary motion we include forces between the planet and its sun, and ignore interactions between planets. In Schelling's model, we include individual decisions based on local information, and ignore every other aspect of human behavior.

But there are differences of degree. For planetary motion, we can defend the model by showing that the forces we ignore are smaller than the ones we include. And we can extend the model to include other interactions and show that the effect is small. For Schelling's model it is harder to justify the simplifications.

Another difference is that Schelling's model doesn't appeal to any physical laws, and it uses only simple computation, not mathematical derivation. Models like Schelling's don't look like classical science, and many people find them less compelling, at least at first. But as I will try to demonstrate, these models do useful work, including prediction, explanation, and design. One of the goals of this book is to explain how.

## The changing criteria of science

Complexity science is not just a different set of models; it is also a gradual shift in the criteria models are judged by, and in the kinds of models that are considered acceptable.

For example, classical models tend to be law-based, expressed in the form of equations, and solved by mathematical derivation. Models that fall under the umbrella of complexity are often rule-based, expressed as computations, and simulated rather than analyzed.

Not everyone finds these models satisfactory. For example, in *Sync*, Steven Strogatz writes about his model of spontaneous synchronization in some species of fireflies. He presents a simulation that demonstrates the phenomenon, but then writes:

> I repeated the simulation dozens of times, for other random initial conditions and for other numbers of oscillators. Sync every time. [...] The challenge now was to prove it. Only an ironclad proof would demonstrate, in a way that no computer ever could, that sync was inevitable; and the best kind of proof would clarify *why* it was inevitable.

Strogatz is a mathematician, so his enthusiasm for proofs is understandable, but his proof doesn't address what is, to me, the most interesting part of the phenomenon. In order to prove that "sync was inevitable", Strogatz makes several simplifying assumptions, in particular that each firefly can see all the others.

In my opinion, it is more interesting to explain how an entire valley of fireflies can synchronize *despite the fact that they cannot all see each other*. How this kind of global behavior emerges from local interactions is the subject of Chapter 9. Explanations of these phenomena often use agent-based models, which explore (in ways that would be difficult or impossible with mathematical analysis) the conditions that allow or prevent synchronization.

I am a computer scientist, so my enthusiasm for computational models is probably no surprise. I don't mean to say that Strogatz is wrong, but rather that people have different opinions about what questions to ask and what tools to use to answer them. These opinions are based on value judgments, so there is no reason to expect agreement.

Nevertheless, there is rough consensus among scientists about which models are considered good science, and which others are fringe science, pseudoscience, or not science at all.

A central thesis of this book is that the criteria this consensus is based on change over time, and that the emergence of complexity science reflects a gradual shift in these criteria.

## The axes of scientific models

I have described classical models as based on physical laws, expressed in the form of equations, and solved by mathematical analysis; conversely, models of complex systems are often based on simple rules and implemented as computations.

We can think of this trend as a shift over time along two axes:

-   Equation-based → simulation-based

-   Analysis → computation

Complexity science is different in several other ways. I present them here so you know what's coming, but some of them might not make sense until you have seen the examples later in the book.

**Continuous → discrete:** Classical models tend to be based on continuous mathematics, like calculus; models of complex systems are often based on discrete mathematics, including graphs and cellular automatons.

**Linear → nonlinear:** Classical models are often linear, or use linear approximations to nonlinear systems; complexity science is more friendly to nonlinear models.

**Deterministic → stochastic:** Classical models are usually deterministic, which may reflect underlying philosophical determinism, discussed in Chapter 5; complex models often include randomness.

**Abstract → detailed:** In classical models, planets are point masses, planes are frictionless, and cows are spherical (see <https://thinkcomplex.com/cow>). Simplifications like these are often necessary for analysis, but computational models can be more realistic.

**One, two → many:** Classical models are often limited to small numbers of components. For example, in celestial mechanics the two-body problem can be solved analytically; the three-body problem cannot. Complexity science often works with large numbers of components and larger numbers of interactions.

**Homogeneous → heterogeneous:** In classical models, the components and interactions tend to be identical; complex models more often include heterogeneity.

These are generalizations, so we should not take them too seriously. And I don't mean to deprecate classical science. A more complicated model is not necessarily better; in fact, it is usually worse.

And I don't mean to say that these changes are abrupt or complete. Rather, there is a gradual migration in the frontier of what is considered acceptable, respectable work. Some tools that used to be regarded with suspicion are now common, and some models that were widely accepted are now regarded with scrutiny.

For example, when Appel and Haken proved the four-color theorem in 1976, they used a computer to enumerate 1,936 special cases that were, in some sense, lemmas of their proof. At the time, many mathematicians did not consider the theorem truly proved. Now computer-assisted proofs are common and generally (but not universally) accepted.

Conversely, a substantial body of economic analysis is based on a model of human behavior called "Economic man", or, with tongue in cheek, *Homo economicus*. Research based on this model was highly regarded for several decades, especially if it involved mathematical virtuosity. More recently, this model is treated with skepticism, and models that include imperfect information and bounded rationality are hot topics.

## Different models for different purposes

Complex models are often appropriate for different purposes and interpretations:

**Predictive → explanatory:** Schelling's model of segregation might shed light on a complex social phenomenon, but it is not useful for prediction. On the other hand, a simple model of celestial mechanics can predict solar eclipses, down to the second, years in the future.

**Realism → instrumentalism:** Classical models lend themselves to a realist interpretation; for example, most people accept that electrons are real things that exist. Instrumentalism is the view that models can be useful even if the entities they postulate don't exist. George Box wrote what might be the motto of instrumentalism: "All models are wrong, but some are useful."

**Reductionism → holism:** Reductionism is the view that the behavior of a system can be explained by understanding its components. For example, the periodic table of the elements is a triumph of reductionism, because it explains the chemical behavior of elements with a model of electrons in atoms. Holism is the view that some phenomena that appear at the system level do not exist at the level of components, and cannot be explained in component-level terms.

We get back to explanatory models in Chapter 4, instrumentalism in Chapter 6, and holism in Chapter 8.

## Complexity engineering

I have been talking about complex systems in the context of science, but complexity is also a cause, and effect, of changes in engineering and the design of social systems:

**Centralized → decentralized:** Centralized systems are conceptually simple and easier to analyze, but decentralized systems can be more robust. For example, in the World Wide Web clients send requests to centralized servers; if the servers are down, the service is unavailable. In peer-to-peer networks, every node is both a client and a server. To take down the service, you have to take down *every* node.

**One-to-many → many-to-many:** In many communication systems, broadcast services are being augmented, and sometimes replaced, by services that allow users to communicate with each other and create, share, and modify content.

**Top-down → bottom-up:** In social, political and economic systems, many activities that would normally be centrally organized now operate as grassroots movements. Even armies, which are the canonical example of hierarchical structure, are moving toward devolved command and control.

**Analysis → computation:** In classical engineering, the space of feasible designs is limited by our capability for analysis. For example, designing the Eiffel Tower was possible because Gustave Eiffel developed novel analytic techniques, in particular for dealing with wind load. Now tools for computer-aided design and analysis make it possible to build almost anything that can be imagined. Frank Gehry's Guggenheim Museum Bilbao is my favorite example.

**Isolation → interaction:** In classical engineering, the complexity of large systems is managed by isolating components and minimizing interactions. This is still an important engineering principle; nevertheless, the availability of computation makes it increasingly feasible to design systems with complex interactions between components.

**Design → search:** Engineering is sometimes described as a search for solutions in a landscape of possible designs. Increasingly, the search process can be automated. For example, genetic algorithms explore large design spaces and discover solutions human engineers would not imagine (or like). The ultimate genetic algorithm, evolution, notoriously generates designs that violate the rules of human engineering.

## Complexity thinking

We are getting farther afield now, but the shifts I am postulating in the criteria of scientific modeling are related to 20th century developments in logic and epistemology.

**Aristotelian logic → many-valued logic:** In traditional logic, any proposition is either true or false. This system lends itself to math-like proofs, but fails (in dramatic ways) for many real-world applications. Alternatives include many-valued logic, fuzzy logic, and other systems designed to handle indeterminacy, vagueness, and uncertainty. Bart Kosko discusses some of these systems in *Fuzzy Thinking*.

**Frequentist probability → Bayesianism:** Bayesian probability has been around for centuries, but was not widely used until recently, facilitated by the availability of cheap computation and the reluctant acceptance of subjectivity in probabilistic claims. Sharon Bertsch McGrayne presents this history in *The Theory That Would Not Die*.

**Objective → subjective:** The Enlightenment, and philosophic modernism, are based on belief in objective truth, that is, truths that are independent of the people that hold them. 20th century developments including quantum mechanics, Gödel's Incompleteness Theorem, and Kuhn's study of the history of science called attention to seemingly unavoidable subjectivity in even "hard sciences" and mathematics. Rebecca Goldstein presents the historical context of Gödel's proof in *Incompleteness*.

**Physical law → theory → model:** Some people distinguish between laws, theories, and models. Calling something a "law" implies that it is objectively true and immutable; "theory" suggests that it is subject to revision; and "model" concedes that it is a subjective choice based on simplifications and approximations.

I think they are all the same thing. Some concepts that are called laws are really definitions; others are, in effect, the assertion that a certain model predicts or explains the behavior of a system particularly well. We come back to the nature of physical laws in Chapter 4 ("Explanatory models"), Chapter 5 ("What is this a model of?") and Chapter 8 ("Reductionism and Holism").

**Determinism → indeterminism:** Determinism is the view that all events are caused, inevitably, by prior events. Forms of indeterminism include randomness, probabilistic causation, and fundamental uncertainty. We come back to this topic in Chapter 5 ("Determinism") and Chapter 10 ("Emergence and free will").

These trends are not universal or complete, but the center of opinion is shifting along these axes. As evidence, consider the reaction to Thomas Kuhn's *The Structure of Scientific Revolutions*, which was reviled when it was published and is now considered almost uncontroversial.

These trends are both cause and effect of complexity science. For example, highly abstracted models are more acceptable now because of the diminished expectation that there should be a unique, correct model for every system. Conversely, developments in complex systems challenge determinism and the related concept of physical law.

This chapter is an overview of the themes coming up in the book, but not all of it will make sense before you see the examples. When you get to the end of the book, you might find it helpful to read this chapter again.

[Think Complexity, 2nd Edition](https://greenteapress.com/wp/think-complexity-2e/)

Copyright 2016 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
