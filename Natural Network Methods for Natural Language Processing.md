## 有趣的问题：

1.如何理解distributed&distributional？二者有什么联系？（chapter10.4.3）

2.如何理解linear模型和attention机制的可解释性？

3.设计一个方法或者模型区分一篇文章是en或者fe写的？从统计思路到深度思路

4.log loss和logistic loss的区别？

5.我们需要语言学知识吗？

6.relation between one-hot representation and dense vector?(in dl, the diffs exist at the first sight.)

7.传统语言模型的缺点是啥？

8.如何高效地解决lm的大词表问题？

9.不同场景下的context是如何定义的？（chpater10.5）

10.distributional representation 有啥缺点？（chapter10.7）

## chapter-12

(1)When learning about a new architecture, don’t think “which existing component does it replace?” or “how do I use it to solve a task?” but rather “how can I integrate it into my arsenal of building blocks, and combine it with the other components in order to achieve a desired result?”.

(2)![img9](http://wx3.sinaimg.cn/mw690/aba7d18bgy1g2dyqh1ed6j220k0e6wq8.jpg)

## chapter-10

1.![img8](http://wx2.sinaimg.cn/mw690/aba7d18bly1g2d0t7feelj21cm0u0gwc.jpg)

## chapter-8(8.4 odds and ends讨论了基础nlp技术的相关细节，精华。)

1.![img5](http://wx3.sinaimg.cn/mw690/aba7d18bgy1g2cp22vr40j21oc0u04b7.jpg)

2.![img6](http://wx4.sinaimg.cn/mw690/aba7d18bgy1g2cp272whfj21z207cjui.jpg)

3.![img7](http://wx3.sinaimg.cn/mw690/aba7d18bly1g2cz0h1234j21ze0kw11c.jpg)

## chapter-7

1.![img3](http://wx1.sinaimg.cn/mw690/aba7d18bly1g2btns5anuj220c0juqbc.jpg)

2.![img4](http://wx1.sinaimg.cn/mw690/aba7d18bly1g2btnmap3gj22120skn7j.jpg)

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
