## 有趣的问题：

1.如何理解distributed&distributional？

2.如何理解linear模型和attention机制的可解释性？

3.设计一个方法或者模型区分一篇文章是en或者fe写的？从统计思路到深度思路

4.log loss和logistic loss的区别？

5.我们需要语言学知识吗？

## chapter-7

1.![img3](http://wx1.sinaimg.cn/mw690/aba7d18bly1g2btns5anuj220c0juqbc.jpg)

2.![imgs](http://wx1.sinaimg.cn/mw690/aba7d18bly1g2btnmap3gj22120skn7j.jpg)

## chapter-5: neural network training

1.derivatives of "non-mathemtical" functions

![img2](http://wx1.sinaimg.cn/mw690/aba7d18bgy1g2b9ghb4uyj223u0kaqf1.jpg)

## chapter-2: learning basics and linear models

1.why do we need regularization?

![img1](http://wx3.sinaimg.cn/mw690/aba7d18bly1g2alqsw9kzj21300hywl9.jpg)

## chapter-1: introduction

11.[distributed&distributional](https://zhuanlan.zhihu.com/p/22386230)

(1) word-context matrix approach

(2) a neural language-modeling inspired word-embedding algs

10. feed-forward neural language model archs.

9.advanced topics: recursive networks for modeling trees, structured prediction models, mtl.

**8.rnn for modeling sequences and stacks.**

7. a series of works managed to obtain imporved syntactic parsing results by simply replacing the linear

model of a parser with a fully connected feed-forward network.

6.structured data of arbitrary sizes: sequences and trees. recurrent/recursive archs will capture

regularities while perserving a lot of the structural information.

5.conv for nlp

a.expect to find strong local clues regarding class membership, but these clues can appear in different

places in the input. We would like to learn that certain sequences of words are good indicators of
 
 the topic.
 
 b.learning informative ngram patterns
 
 **c.hash-kernel as alternative method**
 
 
 
4.mtl&semi-supervised learning

mtl: learning from related problems

semi-supervised: learning from external, unannotated data

3.two important archs in nlp

feed-forward networks(conv: local patterns) and recurrent/recursive networks.

2.view of dl.

a. mathematical view

b.brain-inspired view

1.why nlp is hard?

a. ill-defined and unspecified set of rules

for example, we can't find the function as f(red)=pink, but it's easy to image and others.

b.discrete, compositional(char, word, sentence, paragraph, doc and so on), and sparse

## abstract

1.words representation: vector-based&symbolic representations
