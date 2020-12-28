258.《Metapath-guided Heterogeneous Graph Neural Network for Intent Recommendation》，KDD2019

基于user-item-query构建的图，三种类型的边：search,click, guide，用于淘宝意图检测。

257.《Heterogeneous Graph Attention Network》，WWW2019

异构图+gat，文章写作思路很赞。目前，我们正尝试将该工作用于销量预测与归因分析。

256.《POG：Personalized Outfit Generation for Fashion Recommendation ai Alibaba iFashion》，KDD2019

阿里dida平台建设，compatibility+personality都要考察。

相关PR稿(dida也是从luban演化而来)：https://hackernoon.com/finding-the-perfect-outfit-with-alibabas-dida-ai-assistant-71ba7c9e8cfa

255.《Spam Review Detection with Graph Convolutional Networks》,CIKM2018 Best Paper

重新翻开这篇文章，方法如下：

![img_255](https://ftp.bmp.ovh/imgs/2020/12/42595b8c0cace4ad.png)

文章要解决的问题是垃圾评论检测，构建了两个图。第一个图：用户-评论-商品图，是异构图；第二个图：评论-评论图，是同构图。分别用异构GCN和GCN学到各自的表征，做节点分类工作。

整体上，文章的思路和这篇《Abusive Language Detection with Graph Convolutional Networks》非常相似，但是二者都没有互相引用。这篇文章做的是Tweet分类（三分类），分别构建两个图。第一个图：用户-用户的同构图；第二个图：用户-Tweet的异构图。针对同构图，用node2vec去学到表征（node2vec不是仅仅适用于同构图，不过效果需要考察）；针对异构图，用gcn去学到表征。表征组合（embedding+n-gram）+分类器做节点分类。

对比二者，整体上的技术思路相似，不过显然后者在图构建上更加的自然。

254.《Graph Neural Networks：Taxonomy, Advances and Trends》，最新的GNN相关的综述文章

253.《Understanding Image Retrieval Re-Ranking：A Graph Neural Network Persperctive》

有意思的工作，作者提到：

  （1）Re-ranking can be reformulated as a high-parallelism Graph Neural Network (GNN) function.

  （2）On the Market-1501 dataset, we accelerate the re-ranking processing from 89.2s to 9.4ms with one K40m GPU.

252.《Why Are Deep Learning Models Not Consistently Winning Recommender Systems Competitions Yet?》，RecSys2020

非常棒的文章，多年以前自己就很好奇了。

251.《Enriching Pre-trained Language Model with Entity Information for Relation Classiﬁcation》

![img_251](https://ftp.bmp.ovh/imgs/2020/11/fd20de4ee9f75cfc.png)

250.《Diverse, Controllable, and Keyphrase-Aware:
A Corpus and Method for News Multi-Headline Generation》

新闻标题生成

249.《CharBERT: Character-aware Pre-trained Language Model》

248.《A Frustratingly Easy Approach for Joint Entity and Relation Extraction》

![img248](https://ftp.bmp.ovh/imgs/2020/11/34e4c031463fcc48.jpg)


247.《Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba》

基于Graph的结构，通过Graph Learning的方式获取商品Embedding。这里的Graph是基于历史订单共现数据搭建的（我们自己的搭建方式是基于商品共有属性），整体的思路是类似于Word2Vec。

![img247](https://wx4.sinaimg.cn/mw690/aba7d18bly1gk9s6fxsdhj211a0d679c.jpg)

246.《Iterative Strategy for Named Entity Recognition with Imperfect Annotations》，NLPCC2020的AutoIE赛道的题目

245.《Pretrained Transformers for Text Ranking: BERT and Beyond》，综述文章

244.《Multimodal Attribute Extraction》

现在NLP的多数属性抽取任务是通过直接分析文本的基础上做的，这篇文章在做属性抽取的时候，融合图片的信息。（对多模态不是很了解，很好奇图片的信息真的能够帮助提效吗？）

243.《中文电子病历命名实体和实体关系语料库构建》，2016，软件学报

非常棒的文章。可能不像现在多数数据集构建的文章，会跑一部分的baseline，但是这篇文章的特点更加明显，有用。

（1）总结了诊疗过程中的研究对象

（2）阐述了数据集构建的具体流程和标准等

（3）给出了建模的基本方法

（4）总结了数据构建过程中的经验和教训

值得多看，对其他领域数据集构建也有启发。

242.《MetaPAD: Meta Paern Discovery from Massive Text Corpora》

从大规模语料中挖掘类似于下图左侧的meta pattern。

![img_242](https://wx2.sinaimg.cn/mw690/aba7d18bly1gjmynkys55j20np0gadka.jpg)

241.《Simplify the Usage of Lexicon in Chinese NER》

KeyPoints：如何在做NER的时候，利用lexicon的信息？（Xuanjing Huang老师组的一直在做的Topic）

工作延续：是对Lattice-LSTM的进一步优化，相关工作还有一些，主要解决：在保证metric不变的前提下，提升inference speed。

做法：利用分词（BMEO）+词频（作为权重）

Motivation:分词信息能够完整表征用词典去match句子时得到的匹配信息；Attention是一种dynamic weighting技术，词频是一种静态weighting技术（预计算），类似想法在ZEN的工作中也有体现。

直观感受：一种NER任务的Trick实现。围绕Lattice-LSTM的工作，最近也有一个工作是基于Transformer来做的。

240.《Unsupervised Text Style Transfer with Padded Masked Language Models》

239.《Tuning Word2vec for Large Scale Recommendation Systems》，Twitter

Word2vec用于推荐系统（基于用户行为序列的表征）

238.《What if we had no Wikipedia? Domain-independent Term Extraction from a Large News Corpus》

domain内的自动术语抽取。文章中提到一个有趣的场景：wikipedia的词条哪些是需要编辑的？（人类知识是动态可进化的）

237.《Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks》

讨论BERT系用于domain任务的时候，是否需要进一步pre-training，具体怎么做？在早期，UniLM的工作中讨论过类似做法，去年的一些工作中也有尝试，但是整体结论是：提升是不可预期的，因此个人在后期的一些任务中，多采用直接finetune，没有做进一步的pre-training。

236.《The Renaissance of Templates for Data2Text: A Template Based Data2Text System Powered by Text Stitch Model》，EMNLP2020

这篇想读，但是文章还没有share。

235.《FLAT: Chinese NER Using Flat-Lattice Transformer》,ACL2020

中文NER目前的SOTA，利用分词信息增强BERT，个人比较喜欢的工作，比较干净。黄老师组应该一直在做lexicion-based的NER工作，之前做过PU Learning用于NER的相关工作。[相关参考](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247497102&idx=1&sn=cedddfa134b0a2e0ca30b3f033560eb2&chksm=970fdd58a078544e75a84939682ca36a3dc412c93853f5377d3a17dcc3c5c399567994a34ea1&scene=178#rd)，位置信息的利用采用position-
