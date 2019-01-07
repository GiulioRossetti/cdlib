# Awesome Community Detection

A collection of community detection papers with implementations.


##### Table of Contents  

1. [Factorization](#factorization)  
2. [Deep Learning](#deep-learning) 
3. [Label Propagation, Percolation and Random Walks](#label-propagation-percolation-and-random-walks) 
4. [Tensor Decomposition](#tensor-decomposition)
5. [Spectral Methods](#spectral-methods) 
6. [Temporal Methods](#temporal-methods) 
7. [Cyclic Patterns](#cyclic-patterns)
8. [Centrality and Cuts](#centrality-and-cuts) 
9. [Physics Inspired](#physics-inspired) 
10. [Others](#others) 
  
## Factorization
- **[NO] Graph Embedding with Self-Clustering (Arxiv 2018)**
  - Benedek Rozemberczki, Ryan Davies, Rik Sarkar, and Charles Sutton
  - [[Paper]](https://arxiv.org/abs/1802.03997)
  - [[Python Reference]](https://github.com/benedekrozemberczki/GEMSEC)
  
- **[NO] Deep Autoencoder-like Nonnegative Matrix Factorization for Community Detection (CIKM 2018)**
  - anghua Ye, Chuan Chen, and Zibin Zheng
  - [[Paper]](https://smartyfh.com/Documents/18DANMF.pdf)
  - [[Python Reference]](https://github.com/benedekrozemberczki/DANMF)
  

- **Bayesian Robust Attributed Graph Clustering: Joint Learning of Partial Anomalies and Group Structure (AAAI 2018)**
  - Aleksandar Bojchevski and Stephan Günnemann
  - [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16363/16542)
  - [[Python Reference]](https://github.com/abojchevski/paican)
  
- **Community Preserving Network Embedding (AAAI 17)**
  - Xiao Wang, Peng Cui, Jing Wang, Jain Pei, WenWu Zhu, Shiqiang Yang
  - [[Paper]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14589/13763)
  - [[Python Reference]](https://github.com/benedekrozemberczki/M-NMF)
  
  
- **Semi-supervised Clustering in Attributed Heterogeneous Information Networks (WWW 17)**
  - Xiang Li, Yao Wu, Martin Ester, Ben Kao, Xin Wang, and Yudian Zheng
  - [[Paper]](https://dl.acm.org/citation.cfm?id=3052576)
  - [[Python Reference]](https://github.com/wedoso/SCHAIN-NL)
  
- **Learning Community Embedding with Community Detection and Node Embedding on Graph (CIKM 2017)**
  - Sandro Cavallari, Vincent W. Zheng, Hongyun Cai, Kevin Chen-Chuan Chang, and Erik Cambria
  - [[Paper]](http://sentic.net/community-embedding.pdf)
  - [[Python Reference]](https://github.com/andompesta/ComE)
  
 - **Joint Community and Structural Hole Spanner Detection via Harmonic Modularity (KDD 2016)**
    - Lifang He, Chun-Ta Lu, Jiaqi Mu, Jianping Cao, Linlin Shen, and Philip S Yu
    - [[Paper]](https://www.kdd.org/kdd2016/papers/files/rfp1184-heA.pdf)
    - [[Python Reference]](https://github.com/LifangHe/KDD16_HAM)
      
- **Community Detection via Measure Space Embedding (NIPS 2015)**
  - Mark Kozdoba and Shie Mannor
  - [[Paper]](https://papers.nips.cc/paper/5808-community-detection-via-measure-space-embedding.pdf)
  - [[Python Reference]](https://github.com/komarkdev/der_graph_clustering)
    
- **Overlapping Community Detection at Scale: a Nonnegative Matrix Factorization Approach (WSDM 2013)**
  - Jaewon Yang and Jure Leskovec
  - [[Paper]](http://i.stanford.edu/~crucis/pubs/paper-nmfagm.pdf)
  - [[Python Reference]](https://github.com/RobRomijnders/bigclam)
    
## Deep Learning
- **Improving the Efficiency and Effectiveness of Community Detection via Prior-Induced Equivalent Super-Network (Scientific Reports 2017)**
  - Liang Yang, Di Jin, Dongxiao He, Huazhu Fu, Xiaochun Cao, and Francoise Fogelman-Soulie
  - [[Paper]](http://yangliang.github.io/pdf/sr17.pdf)
  - [[Python Reference]](http://yangliang.github.io/code/SUPER.zip)
  
- **Community Detection with Graph Neural Networks (ArXiv 2017)**
  - Zhengdao Chen, Xiang Li, and Joan Bruna
  - [[Paper]](https://arxiv.org/abs/1705.08415)
  - [[Python Reference]](https://github.com/afansi/multiscalegnn)
  
- **Modularity based Community Detection with Deep Learning (IJCAI 2016)**
  - Liang Yang, Xiaochun Cao, Dongxiao He, Chuan Wang, Xiao Wang, and Weixiong Zhan
  - [[Paper]](http://yangliang.github.io/pdf/ijcai16.pdf)
  - [[Python Reference]](http://yangliang.github.io/code/DC.zip)
  
- **Learning Deep Representations for Graph Clustering (AAAI 2014)**
  - Fei Tian, Bin Gao, Qing Cui, Enhong Chen, and Tie-Yan Liu
  - [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8527)
  - [[Python Reference]](https://github.com/quinngroup/deep-representations-clustering)
  
## Label Propagation, Percolation and Random Walks
- **[DONE] Multiple Local Community Detection (ACM SIGMETRICS 2017)**
  - Alexandre Hollocou, Thomas Bonald, and Marc Lelarge
  - [[Paper]](https://hal.archives-ouvertes.fr/hal-01625444)
  - [[Python Reference]](https://github.com/ahollocou/multicom)
 
- **[DONE] SLPA: Uncovering Overlapping Communities in Social Networks via A Speaker-listener Interaction Dynamic Process (ICDMW 2011)**
  - Jierui Xie, Boleslaw K Szymanski, and Xiaoming Liu
  - [[Paper]](https://arxiv.org/pdf/1109.5720.pdf)
  - [[Python Reference]](https://github.com/kbalasu/SLPA)

- **[DONE] The Map Equation (The European Physical Journal Special Topics 2009)**
  - Martin Rossvall, Daniel Axelsson, and Carl T Bergstrom
  - [[Paper]](https://arxiv.org/abs/0906.1405)
  - [[Python Reference]](https://github.com/Tavpritesh/MapEquation)
  
- **Community Detection by Information Flow Simulation (ArXiv 2018)**
  - Rajagopal Venkatesaramani and Yevgeniy Vorobeychik 
  - [[Paper]](https://arxiv.org/abs/1805.04920)
  - [[Python Reference]](https://github.com/rajagopalvenkat/Community_Detection-Flow_Simulation)
  
- **Modeling Community Detection Using Slow Mixing Random Walks (IEEE Big Data 2015)**
  - Ramezan Paravi, Torghabeh Narayana, and Prasad Santhanam
  - [[Paper]](https://ieeexplore.ieee.org/abstract/document/7364008)
  - [[Python Reference]](https://github.com/paravi/MarovCommunity)
    
- **Overlapping Community Detection Using Seed Set Expansion (CIKM 2013)**
  - Joyce Jiyoung Whang, David F. Gleich, and Inderjit S. Dhillon
  - [[Paper]](http://www.cs.utexas.edu/~inderjit/public_papers/overlapping_commumity_cikm13.pdf)
  - [[Python Reference]](https://github.com/pratham16/community-detection-by-seed-expansion)
  
- **Chinese Whispers: an Efficient Graph Clustering Algorithm and its Application to Natural Language Processing Problems (HLT NAACL 2006)**
  - Chris Biemann
  - [[Paper]](http://www.aclweb.org/anthology/W06-3812)
  - [[Python Reference]](https://github.com/sanmayaj/ChineseWhispers)
  
- **An Efficient Algorithm for Large-scale Detection of Protein Families (Nucleic Acids Research 2002)**
  - Anton Enright, Stijn Van Dongen, and Christos Ouzounis
  - [[Paper]](https://academic.oup.com/nar/article/30/7/1575/2376029)
  - [[Python Reference]](https://github.com/HarshHarwani/markov-clustering-for-graphs)
  - [[Python Reference]](https://github.com/lucagiovagnoli/Markov_clustering-Graph_API)
  
## Tensor Decomposition
  
- **Overlapping Community Detection via Constrained PARAFAC: A Divide and Conquer Approach (ICDM 2017)**
  - Fatemeh Sheikholeslami and Georgios B. Giannakis 
  - [[Paper]](https://ieeexplore.ieee.org/document/8215485)
  - [[Python Reference]](https://github.com/FatemehSheikholeslami/EgoTen)
  
## Spectral Methods
  
- **Local Lanczos Spectral Approximation for Community Detection (ECML PKDD 2017)**
  - Pan Shi, He Kun, David Bindel, and John Hopcroft
  - [[Paper]](http://ecmlpkdd2017.ijs.si/papers/paperID161.pdf)
  - [[Python Reference]](https://github.com/PanShi2016/LLSA)

- **Spectral Clustering with Graph Filtering and Landmark Based Representation (ICASSP 2016)**
  - Nicolas Tremblay, Gilles Puy, Pierre Borgnat, Rémi Gribonval, and Pierre Vandergheynst
  - [[Paper]](https://arxiv.org/pdf/1509.08863.pdf)
  - [[Python Reference]](https://github.com/cylindricalcow/FastSpectralClustering)

- **Uncovering the Small Community Structure in Large Networks: a Local Spectral Approach (WWW 2015)**
  - Li Yixuan, He Kun, David Bindel, and John Hopcroft
  - [[Paper]](https://arxiv.org/abs/1509.07715)
  - [[Python Reference]](https://github.com/YixuanLi/LEMON)
  
## Cyclic Patterns

- **Adaptive Modularity Maximization via Edge Weighting Scheme (Information Sciences 2018)**
  - Xiaoyan Lu, Konstantin Kuzmin, Mingming Chen, and Boleslaw K Szymanski
  - [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-72150-7_23)
  - [[Python Reference]](https://github.com/xil12008/adaptive_modularity)
  
- **Graph sketching-based Space-efficient Data Clustering (SDM 2018)**
  - Anne Morvan, Krzysztof Choromanski, Cédric Gouy-Pailler, Jamal Atif
  - [[Paper]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975321.2)
  - [[Python Reference]](https://github.com/annemorvan/DBMSTClu)
  
- **GMAC: A Seed-Insensitive Approach to Local Community Detection (DaWak 2013)**
  - Lianhang Ma, Hao Huang, Qinming He, Kevin Chiew, Jianan Wu, and Yanzhe Che
  - [[Paper]](https://link.springer.com/chapter/10.1007/978-3-642-40131-2_26)
  - [[Python Reference]](https://github.com/SnehaManjunatha/Local-Community-Detection)
  
## Centrality and Cuts
  
- **Real-Time Community Detection in Large Social Networks on a Laptop (PLOS 2018)**
  - Benjamin Paul Chamberlain, Josh Levy-Kramer, Clive Humby, and Marc Peter Deisenroth
  - [[Paper]](https://arxiv.org/pdf/1601.03958.pdf)
  - [[Python Reference]](https://github.com/melifluos/LSH-community-detection)
  
- **A Community Detection Algorithm Using Network Topologies and Rule-based Hierarchical Arc-merging Strategies (PLOS 2018)**
  - Yu-Hsiang Fu, Chung-Yuan Huang, and Chuen-Tsai Sun
  - [[Paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0187603)
  - [[Python Reference]](https://github.com/yuhsiangfu/Hierarchical-Arc-Merging)
  
- **Query-oriented Graph Clustering (PAKDD 2017)**
  -  Li-Yen Kuo, Chung-Kuang Chou, and Ming-Syan Chen
  - [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-57529-2_58)
  - [[Python Reference]](https://github.com/iankuoli/QGC)
  
- **Graph Degree Linkage: Agglomerative Clustering on a Directed Graph (ECCV 2012)**
  - Wei Zhang, Xiaogang Wang, Deli Zhao and Xiaoou Tang
  - [[Paper]](https://arxiv.org/abs/1208.5092)
  - [[Python Reference]](https://github.com/myungjoon/GDL)
  
## Physics Inspired
- **[DONE]Fluid Communities: A Community Detection Algorithm (Complenet 2017)**
  - Ferran Parés, Dario Garcia-Gasulla, Armand Vilalta, Jonatan Moreno, Eduard Ayguadé, Jesús Labarta, Ulises Cortés and Toyotaro Suzumura
  - [[Paper]](https://arxiv.org/abs/1703.09307)
  - [[Python Reference]](https://github.com/HPAI-BSC/Fluid-Communities)

- **Thermodynamics of the Minimum Description Length on Community Detection (ArXiv 2018)**
  - Juan Ignacio Perotti, Claudio Juan Tessone, Aaron Clauset and Guido Caldarelli
  - [[Paper]](https://arxiv.org/pdf/1806.07005.pdf)
  - [[Python Reference]](https://github.com/jipphysics/bmdl_edm)

  
- **Defining Least Community as a Homogeneous Group in Complex Networks (Physica A 2015)**
  - Renaud Lambiotte, J-C Delvenne, and Mauricio Barahona
  - [[Paper]](https://arxiv.org/pdf/1502.00284.pdf)
  - [[Python Reference]](https://github.com/dingmartin/HeadTailCommunityDetection)
  
- **Think Locally, Act Locally: Detection of Small, Medium-Sized, and Large Communities in Large Networks (Physica Review E 2015)**
  - Lucas G. S. Jeub, Prakash Balachandran, Mason A. Porter, Peter J. Mucha, and Michael W. Mahoney
  - [[Paper]](https://arxiv.org/abs/1403.3795v1)
  - [[Python Reference]](https://github.com/LJeub/LocalCommunities)
    
## Others

- **General Optimization Technique for High-quality Community Detection in Complex Networks (Physical Review E 2014)**
  - Stanislav Sobolevsky, Riccardo Campari, Alexander Belyi, and Carlo Ratti
  - [[Paper]](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.012811)
  - [[Python Reference]](https://github.com/Casyfill/pyCombo)
  
- **Community Detection via Maximization of Modularity and Its Variants (IEEE TCSS 2014)**
  - Mingming Chen, Konstantin Kuzmin, and Boleslaw K. Szymanski 
  - [[Paper]](https://www.cs.rpi.edu/~szymansk/papers/TCSS-14.pdf)
  - [[Python Reference]](https://github.com/itaneja2/community-detection)

- **A Smart Local Moving Algorithm for Large-Scale Modularity-Based Community Detection (The European Physical Journal B 2013)**
  - Ludo Waltman and Nees Jan Van Eck
  - [[Paper]](https://link.springer.com/content/pdf/10.1140/epjb/e2013-40829-0.pdf)
  - [[Python Reference]](https://github.com/chen198328/slm)
