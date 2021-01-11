263.《Read, Retrospect, Select: An MRC Framework to Short Text Entity Linking》

和《AutoRegressive Entity Retrieval》，ICLR2021一块儿读。

262.《CRSLab：An Open-Source Toolkit for Building Conversational Recommender System》

CRS系统的设定：

![img_262](https://ftp.bmp.ovh/imgs/2021/01/d3be7c25a4f1419c.png)


261.《Open Domain Event Extraction Using Neural Latent Variable Models》

开放域的事件抽取。（个人对隐变量模型不是很了解）

260.《Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba》，KDD2018

![260_1_img](https://ftp.bmp.ovh/imgs/2021/01/db47ef98a5f065f4.png)

![260_2_img](https://ftp.bmp.ovh/imgs/2021/01/fbc61749b4d401cf.png)

259.《AutoRegressive Entity Retrieval》，ICLR2021

用生成的方式（seq2seq）做el，entity disambiguation， page retrieval任务。将传统分类任务转化为一个生成任务是问题解决范式的转变，很有意思的工作。在自己的博客，[MRC is all you need?](https://zhpmatrix.github.io/2020/05/07/mrc-is-all-you-need/)中讨论了将很多经典NLP任务用MRC的方式来做。

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

248.《
