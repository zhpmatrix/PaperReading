61.《PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization》

提示：共52页，附录40页。方法如下：

![img](https://wx1.sinaimg.cn/mw690/aba7d18bly1gatvxcslymj20oj0bgwgu.jpg)

60.《Attention Is All You Need》（Transformer的概念难道不是在这里定义的吗？为啥好多paper瞎写，好多人瞎说，不是只有encoder啊。）

亮点：

（1）SOTA
（2）并行(feed forward是position wise的；self-attention的矩阵乘法计算；multi-head等)
（3）简单（抛弃了recurrence和conv）

+ Attention: 在建模input和output之间的dependency时，无需考虑二者之间的distance。dependency不一定必须是distance的函数，取决于如何定义dependency?

描述： output = f(a query, a set of key-value pairs)

为啥选择dot-product attention而不是additive attention？**最主要的目的还是工程上快。因为存在计算更快，空间更加高效的矩阵乘法。**二者的理论时间复杂度一样，实际上，后者的效果不依赖于待计算vector的维度大小，且整体上效果更好。但是还想用前者，那就必须消除vector的维度影响，因此分母除sqrt(d\_k)。实际上，对前者，V的weight是直接计算得到，而对后者，是learn到的。那么，为啥d\_k会对前者产生影响？主要是qk的variance是与d\_k有关的，当d\_k很大时，variance很大，这样容易导致gradient进入softmax的饱和区，玩儿个锤子。

Transformer的优点？（思考角度）

（1）每层的计算复杂度

（2）可并行的总量

（3）能否很好地learn到long range dependency（很遗憾，这里dependency仍旧是distance的函数）

（4）可解释性

+ 时间复杂度：关联input和output任意两个positon的信号，需要的操作的次数？Transformer=O(constant),ConvS2S=O(N),ByteNet=O(logN,不是很确定),这里的区别和用数组还是用链表相似。

+ Memory Network是基于recurrent attention机制的，不是sequence-aligned recurrence（类似2014年经典的seq2seq+attention），从这点儿来讲，Transformer也算是延续了Memory Network的血脉，尤记得当年Memory Network🔥过。

+ position encoding

两种方式如下：

（1）fixed（相对编码）

（2）learnable

结论：效果差不多。但是learnable的需要预先learn到才能用（**BERT，长度512**），但是fixed版就是pos的函数，可以直接计算，因此可以扩展到任意input长度序列。

+ Transformer定义:

>  The Transformer follows encoder-decoder structure using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

Decoder端的三个细节：

+ Masked Multi-Head Attention

对于“我 爱 吃 苹 果 。”， 预测“吃”这个token的时候，不能看到“苹果。”这三个token。

+ shifted right

训练时：输入“<sos> 我 爱 吃 苹 果 。”， 预测“我 爱 吃 苹 果 。<eos>”

预测时（自回归）：生产者-消费者模型，一次一个。

+ 共享

encoder端和decoder端的embedding层共享，pre-softmax linear层共享。这里有意思的点儿是，**将一些机制用于预训练seq2seq模型中？**

三种Regularization策略：

（1）每个sub-layer的输出

（2）sums of the embeddings

（3）label smoothing:虽然会hurt到ppl（该trick的目的就是使得模型变得unsure），但是可以提高bleu。

其他Trick：

（1）average last few checkpoints

（2）基于devset，找到beam size值和length penalty值

（3）maximum output length = input length + 50，允许早停

59.《GPT-based Generation for Classical Chinese Poetry》

整体思路如下：

![img59](https://wx1.sinaimg.cn/mw690/aba7d18bly1gana6e98duj20r30g140q.jpg)

该思路可以对比中文纠错的两种思路：

（1）seq2seq

（2）序列标注（自编码）

对于对联，诗等的生成，（1）是经典思路，（2）是本文的思路（自回归）。用序列标注的优点之一在于：平行数据是天然对齐的。此外，两个问题：

（1）自编码可行？（针对对齐场景）

（2）中文维基的预训练 vs 古诗词的预训练（两种语言结构）

最后，看到文章写道：

> Though the generated poems are not perfect all the time, our preliminary experiments have shown that GPT provides a good start to promote the overall quality of generated poems.

心头一惊。


58.《Looking Beyond Label Noise: Shifted Label Distribution Matters in Distantly Supervised Relation Extraction》

没有认真读，这里藏着，主要讨论：一个经典关系抽取数据集中的数据bias对metric的影响。

类似工作：[人脸识别数据集的身份重合问题](https://zhuanlan.zhihu.com/p/31968327)

其他还有一些，记不起来了。

57.两个有意思的经验性观点：

+ 《Deep Double Descent: Where Bigger Models and More Data Hurt》

> By considering larger function classes, which contain more candidate predictors compatible with the data, we are able to ﬁnd interpolating functions that have smaller norm and are thus “simpler”. Thus increasing function class capacity improves performance of classiﬁers.

参考如下图：

![57_img](https://wx1.sinaimg.cn/mw690/aba7d18bly1gaih6g6q7kj20tz0dywgw.jpg)

+ 《The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks》

彩票假设：

> A randomly initialized, dense neural network contains a subnetwork that is initialized such that when trained in isolation, it can match the test accuracy of the original network after training for at most the same number of iterations.

彩票假设的启发：

（1）pruning irrelevant weight

（2）retrain from scratch using only the 'lottery ticket' weights

56.《K-BERT: Enabling Language Representation with Knowledge Graph》,AAAI2020

融合KG的信息，啥也不想说了。

55.《PRETRAINED ENCYCLOPEDIA: WEAKLY SUPERVISED KNOWLEDGE-P RETRAINED LANGUAGE MODEL》,iclr2020

这篇文章通过设计预训练任务，希望guide模型去learn到更多的fact。整体思路是：对于一句话，找到话中的实体，用其他类型相同，但是内容不同的实体替代，然后训练模型，训练目标是多任务形式，实体有没有被替换+MLM。至于这样做的原因，文章中提到：

> Compared to the language modeling objective, entity replacement is deﬁned at the entity level and introduces stronger negative signals. When we enforce entities to be of the same type, we pre- serve the linguistic correctness of the original sentence while the system needs to learn to perform judgment based on the factual aspect of the sentence.

类似思想其实可以用在很多其他地方。

相关文章：《Language Models as Knowledge Bases?》

54.《Probing sentence embeddings for linguistic properties》，ACL2018

属于Probing类工作，讨论句子embedding和语言学属性之间的关系。设计的三个层面的probing任务值得关注一下（但是个人并不认为这些能够完整体现语言学属性。另一方面，此类任务的设计似乎并不是一个简单的任务）：

+ surface information:句子长度等

+ syntactic information:语法树的深度等

+ semantic information: 词的时态

53.《Learning Sparse Sharing: Architectures for Multiple Tasks》, AAAI2020

类似思想体现在很多地方。文章中的一张图有意思：

![img53](https://mmbiz.qpic.cn/mmbiz_png/vJe7ErxcLmgUZb0kT1OmH0CNKpqTd77uMiazADJHiauicTMoia6LmkjtZNDUic1GF1vnU9syIYzibKicUSej07cM60p2Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)


52.《Multi-channel Reverse Dictionary Model》,AAAI2020

解决的问题：给定对词的描述，返回对应的词。

整体思路：对词的描述进行encode，然后和词的embedding算距离。因此，需要好的对词的描述进行特征抽取的encoder。

文章工作：加feature。（对于multi-channel相关的，多是加feature）

想法：不谈novelty，解决的问题很有意思。对于文章对应的系统，可以爬一爬，做weak-supervision相关。

51.《Likelihood Ratios for Out-of-Distribution Detection》

略读。问题：对OOD样本的处理。

想法：训练模型时，符合IID的假设；但是当模型部署到线上的时候，有时候很难保证IID能够满足，首先**不知道什么时候会出现OOD**，比如测试数据会随着时间变化。**如果出现OOD，这个时候如何处理OOD呢？**

（1）对OOD赋予一个较低的置信度，模型对待OOD的样本要保守。但是，有时候模型不会这样表现，这又是另外一个有趣的问题了；

（2）给定一个样本，判断是否是OOD?这个可以做二分类，可以训练一个生成模型，可以转化为一个outlier detection任务等；

NeurIPS2019，Bengio单独讲了一节，从IID->OOD，希望之后的工作从1-型系统转向2-型系统，也就是从感知层转向推理层，推理一定程度上是彻底解决OOD的方法之一。

50.Contrastive Learning

《Representation Learning with Contrastive Predictive Coding》，deepmind，提出了一个统一的框架。

《DATA-EFFICIENT IMAGE RECOGNITION WITH CONTRASTIVE PREDICTIVE CODING》，deepmind，这篇是上篇主要在image中的应用，做了一些改变，更偏应用。

《Momentum Contrast for Unsupervised Visual Representation Learning》，将contrastive learning的想法用于image领域的无监督预训练。

49.《DATA-DEPENDENT GAUSSIAN PRIOR OBJECTIVE FOR LANGUAGE GENERATION》，iclr2020

主要工作：在传统文本生成的loss函数（mle）中，添加一个kl term，这个term是基于gaussian，data-dependent的，主要动机是考虑那些非max prob的prob的分布，这些prob之间的信息同样是非常丰富的哦。举个简单例子：

猫，猪，苹果三分类任务中，对猫的预测结果是（0.7，0.2，0.1），分别对应猫，猪，苹果，这个预测是正确的，但是（0.2，0.1）也是有信息结果的，这个prob向量意味着猫和猪的距离比猫和苹果的距离近啊。

想到了什么？知识蒸馏呗，同样包括一些针对点，用一个分布来modeling的一些工作，好像在一些regression模型中有用。怎么挖掘这些信息，用一个分布来表达，这里的问题是：用啥分布？怎么得到分布的参数？（learnable）

除此之外，该工作融合了prior的信息，实验证明，low resource的setting下，提升明显；但是数据量大的时候，效果变弱。不管怎样，将prior信息融合model，是一个有趣的方向，可做的事情应该挺多的。

整体上，文章的实验，分析和动机都齐全，novelty倒也不是特别大，但是仍旧是一篇好文。东南大学的耿鑫老师做的label distribution learning其实也挺有意思，不过没有深挖。

48.《HOW MUCH POSITION INFORMATION DO CONVOLUTIONAL NEURAL NETWORKS ENCODE ?》，iclr2020

结论：**surprising degree of absolute position information that is encoded in commonly used neural networks.**

47.《Sequence Modeling with Unconstrained Generation Order》,NeurIPS 2019

seq2seq中，decoder不一定都要按照left-to-right的顺序，也可以任意方向。主要的工作集中在：

（1）允许任意方向：去掉mask

（2）如何定义方向？用position

（3）如何learn？token+position同时learn

有点反直觉？对啦。来来来，先写第三个字，再写第二个字，再写第八个字...无论怎样，从理论上讲，exposure bias应该会缓解。开个脑洞：基于bert，用sequence lableling的思路来做生成，解决output端和input端不需要严格对齐的问题。总之，memory bank也不要了吧。

46.《Mixtape：Breaking the Softmax Bottleneck Efficiently》，NeurIPS 2019

基于一个假设：**Softmax瓶颈就是说模型的低秩性无法充分表达高秩语言空间**，衍生出的两个方法Mos和Mixtape。其中Mos是指多个softmax的混合, Mixtape在效果上和Mos基本一致，但是大大提升了速度，因为本质上在保证高秩的同时，相比于Mos需要计算多次softmax，同时保存中间logit，在存储和速度上都有改善。

需要思考的点儿：

（1）假设本身的合理性？

（2）“信息冗余”的思想在一个侧面上的多种实现？

类似工作：各种各样的weight normalization实现......逃。

45.《“爱情像数学一样复杂”：用于社交聊天机器人的比喻生成系统》

最近比较关注word2vec的更多的应用，相关工作包括这篇文章，大词林的工作，**Trans系列（还没系统想过）**。

这篇文章主要解决给定一个本体，生成一个基于本体的比喻句。整体分为三个工作：

（1）找喻体

（2）找连接词

（3）套模版

这件事情能做的本质原因：本体往往来自抽象域，而喻体往往来自具体域。换言之，人们设法利用比喻，用容易理解的具体概念（即喻体）来解释和表达不易理解的抽象概念（即本体）。（通俗点说：你咋说都是对的。其实，古诗词也是这样滴，古诗词的韵味在于诗词本身和读者之间的交互，具体比如通过联renwu想的方式等）
33.《Image Classiﬁcation with Deep Learning in the Presence of Noisy Labels: A Survey》

44.《NEZHA: Neural Contextualized Representation for Chinese Language Understanding》,[相关介绍](https://mp.weixin.qq.com/s/RkCLSRyy_GuLOXMVSzTMdA)

改进维度：

模型：相对位置编码的利用（Functional Relative Positional Encoding，不需要learn。）

预训练任务：词级Mask+Span信息的利用

训练算法：混合精度+适用于大Batch(300000)的优化器LAMB

在训练过程中，我们采用混合精度训练（Mixed Precision Training）方式，在传统的深度学习训练过程中，所有的变量包括weight，activation和gradient都是用FP32（单精度浮点数）来表示。而在混合精度训练过程中，每一个step会为模型的所有weight维护一个FP32的copy，称为Master  Weights，在做前向和后向传播过程中，Master Weights会转换成FP16（半精度浮点数）格式，权重，激活函数和梯度都是用FP16进行表示，最后梯度会转换成FP32格式去更新Master Weights。

Ablation Study: 位置编码，masking策略，span预测任务，训练序列的长度，训练语料的大小均能带来提升，其中位置编码会带来显著提升；

**最大的收获：位置编码的改进。**

43.《DEEP ENSEMBLES:A LOSS LANDSCAPE PERSPECTIVE》

**Why do ensembles trained with just random initialization work so well in practice?**

42.《Learning From Positive and Unlabeled Data: A Survey》

相关：《A Survey on Postive and Unlabelled Learning》

41.《Distantly Supervised Named Entity Recognition using Positive-Unlabeled Learning》

40.《150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com》,KDD2019, Applied Data Science Track

从Booking.com发布的150个机器学习模型，总结了6条经验，并提供了一些case分析。比如：提高模型的性能并不一定会转化为业务价值的增长；一些最强大的改进并非是在给定setting的上下文中改进模型，而是更改setting本身；减少预测服务的延迟（模型冗余/稀疏模型/预计算和缓存/请求打包/尽可能少的特征变换），尽早获得关于模型质量的反馈（响应分布分析）等

相关文章：

（1）[《Machine Learning:The High-Interest Credit Card of Technical Debt》](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43146.pdf)

（2）[《Hidden Technical Debt in Machine Learning Systems》](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)

（3）[Github: Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)

39.《Open Domain Web Keyphrase Extraction Beyond Language Modeling》，EMNLP2019

开放域的关键短语抽取。主要贡献：

（1）做了一个目前可能最大的开放域的英文短语抽取数据集。

（2）**提出基于网页搜索的弱监督方案。**（目前的网页搜索，是基于纯字符串匹配，作为候选页面，很早自己就有这个想法，其实想办法用起来还是很酷的。）

（3）提出利用visual feature做增强，可能和数据集构建有关吧，个人不是很感兴趣。

（4）做了一个有点丑陋的架构（基于预训练语言模型）。

难点：当要去learn一个非常general的概念的时候，可能是hard to learn的。

38.《Generating Abstractive Summaries with Finetuned Language Models》

Alexander M. Rush组的工作，讨论预训练语言模型用于文本生成，特别是在seq2seq架构下。目前有很多人在关注这个方向，包括自己组里也在做一些探索，不过效果不是特别好。这篇文章，大致是decoder端用了一个预训练语言模型，不过取得的结果<=transformer+copy在摘要任务上的效果。

37.《TENER: Adapting Transformer Encoder for Named Entity Recognition》

魔改模型上有启发。文章不和BERT比，其实感觉对比意义也不是很大。如果基于此结构，做pretrain，会不会进一步提升BERT在NER任务上的表现？虽然文章讲了Transformer在NER任务上表现不好的原因，但是很好奇是否是真的不好？（这个需要更多实验结果的支持。在一个善于讲故事的年代，总得发生点什么才显得可信。）

36.《A Deep Look into Neural Ranking Models for Information Retrieval》

一篇关于Ranking的综述文章。hinge loss可以用在pairwise的ltr任务中。

35.《PyTorch: An Imperative Style, High-Performance Deep Learning Library》，NIPS2019

讨论系统哲学的学术文章，其实官方博客已经讨论了很多相关思想了，所以这篇作为学术文章可能就是方便大家引用吧。作为宇宙最屌框架，PyTorch已经不需要用一篇文章的引用量来证明自己了。

34.《Using Knowledge Graphs for Fact-Aware Language Modeling》，ACL2019

相关的想法：

（1）bert系通过mlm能否直接fact-aware？已经有人做了一些探索。

（2）没有没关系。把kg的信息融入encoder端，提升decoder的能力，就是这篇了。

（3）结合各个任务，融合kg的信息，那么怎么融合kg的信息，就是一个可以玩儿出花儿来的事情。

33.《Spam Review Detection with Graph Convolutional Networks》，CIKM2019最佳应用论文

将GCN用于闲鱼的垃圾营销广告检测。特征提取层上分为两个部分，分别是基于商品-用户-评论的二部图和不同评论构成的图，GCN作为特征提取器。分类层采用TextCNN。

32.《Chinese Word Segmentation as Character Tagging》，2003年的文章，引用量很高。

好像是第一次将分词问题modeling为一个tagging问题来做，共有四个标签（LL/MM/RR/LR）如下：

![img_word_segmentation](https://wx3.sinaimg.cn/mw690/aba7d18bly1g9cvr16rhwj20t808t40d.jpg)

31.《How to Ask Better Questions? A Large-Scale Multi-Domain Dataset for Rewriting Ill-Formed Questions》

做了一个问题改写的数据集。

30.《Learning Semantic Hierarchies via Word Embeddings》，ACL2014

《大词林》，用word2vec的思路做“上下位”关系挖掘。word2vec用好了，应该可以解决很多问题，包括embedding，相关性等任务。

用于开放域的实体类别层次化（上下位关系）。核心思想是用词向量之间的差值刻画上下位关系。工作分为三个步骤：词向量学习，映射矩阵学习和上下位关系判断。上下位关系的判断也可归于关系抽取任务，是一种特殊的较为抽象的关系。因此可以很自然地将文章的想法拓展到关系分类任务上，对每类关系学习一个映射矩阵。当用于SPO三元组直接抽取时，需要有针对性的映射关系学习方法。

29.《Statistical Machine Translation: IBM Models 1 and 2》，Michael Collins

Noisy Channel Model的经典案例：作为统计模型用于机器翻译任务。除此之外，可以用于拼写纠错，Auto Suggestion等，大二时实现的Bayes Matting也有类似的感觉。总之，NCM是一个比较general的理论模型。

28.《Few-Shot Sequence Labeling with Label Dependency Transfer and Pair-wise Embedding》

 Few-Shot Learning的工作，用于命令实体识别任务上。

27.《Hierarchically-Reﬁned Label Attention Network for Sequence Labeling》,EMNLP2019

Label Embedding+Attention用于sequence labeling。平均提升了不足一个百分点，可能和相关任务原来的指标已经较高有关系。不过工作做的还是比较干净的。

26.《Multi-instance Multi-label Learning for Relation Extraction》

2012年的工作，引用量400+。将distant-supervision得到的数据，建模为一个miml问题。主要技术：graphical model+EM。文章引用了周志华老师的miml的相关工作。

延伸思考：

（1）snorkel的相关技术可以结合distant-supervision，比如denoising。

（2）更加general的relation extraction可以关注fewrel的进展。

（3）distant-supervision需要关注multi-instance的相关工作。

相关工作：《Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks》

25.《Distant supervision for relation extraction without labeled data》

2009年的工作，引用非常多，但是并非distant-supervision第一次提出，作者之一是Dan Jurafsky。文章在关系抽取的时候，还是基于人工构建的feature，包括syntactic和lexicial feature等。

相关启发：

（1）regular expr是learnable的；instance和pattern的snowball式玩法。

（2）distant-supervision下，multi-instance的利用是亮点。

24.《KG-BERT: BERT for Knowledge Graph Completion》

这篇工作采用了和之前事件判别模型类似的思路，不同之处在于直接基于三元组做，不包含上下文的具体描述，在相关数据集上取得了SOTA结果。虽然作者给出了一些解释，不过文章实验做的不够充足，并且解释似乎不是特别具有说服性。

结论：灌水。

23.**如何使用更多的外部数据提升模型效果？**

（1）《Exploiting Monolingual Data at Scale for Neural Machine Translation》,EMNLP2019

观点：在机翻中，单独地利用src端做Forward Translation和单独地利用tgt端做Back Translation，在模型性能提升到一定阶段后都会有损失，不过二者的损失速度不同。那么，这篇文章综合利用二者提升机翻的效果。

整体上数据利用流程：标注的平行语料->混合语料（标注+伪标签+**加噪**）->伪标签语料。

过程中需要训练/微调的翻译模型个数：8个。

需要强调的是：加噪很重要。

（2）《Self-training with Noisy Student improves ImageNet classification》，两天前的文章

贡献：标准的self-training的范式，将ImageNet的分类指标极限又推进了一点儿。

需要强调的是：student的学习需要加噪。原文如下：

During the generation of the pseudo labels, the teacher is not noised so that the pseudo labels are as good as possible. But during the learning of the student, we inject noise such as data augmentation, dropout, stochastic depth to the stu- dent so that the noised student is forced to learn harder from the pseudo labels. 

上述两篇文章的延伸思考：

  当数据是构造得到的时候，加噪对下游模型的学习可能会是比较重要的一个因素。除了上述两篇文章，类似的观点在ELECTRA的工作中也可以看到是一个非常重要的地方。在一些场景下，数据的构造过程是比较困难的任务。因为，当开始构造数据的时候，其实已经引入了一定的归纳偏置。这里包括：平衡性，难易程度以及与任务强相关的一些信号等。总之，当模型是在自己构建的数据上去learn的时候，对模型在真实数据集上的表现保持警惕应该没有错，本质上还是i.i.d的问题。

22.《ZEN: Pre-training Chinese Encoder Enhanced by N-gram Representations》

号称目前为止最强中文NLP预训练模型。整体思路上可以从deep&wide结构来理解。deep结构和传统的bert类似，wide结构用来encode n-gram的信息。其实，提供了缓解中文bert在词语粒度上modeling不足的问题。

延伸思考：对词语粒度的语义单元modeling是关于中文bert的老问题了。一般有两个思路：第一个思路，直接在deep结构的input端添加embedding信息，但是这样的问题在于可能会引入noise信息；第二个思路，使用wide结构，单独训练词语粒度的embedding，最后在output端进行信息的融合。这里选择了第二种，其实也是一种比较general的思路。

虽然不清楚在哪个粒度建模比较合适，但是从最近的一些工作来看，融合一些高层semantic信息（多粒度的信息）不是一件很坏的事情。不管怎样，整体上是我喜欢的思路，简单有效。

具体图如下：

![img_zen](https://wx4.sinaimg.cn/mw690/aba7d18bly1g8om2b1jsej20iv0gradb.jpg)

21.《ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS》,ICLR2020

主要内容：Thorough experiments demonstrate this new pre-training task is more efﬁ- cient than MLM because the model learns from all input tokens rather than just the small subset that was masked out.

延伸思考：和non-parallel style transfer的工作类似。不过这篇的主要目的是学到一个好的MLM。图示如下：

![img_electra](https://wx3.sinaimg.cn/mw690/aba7d18bly1g8jwz090qcj20
