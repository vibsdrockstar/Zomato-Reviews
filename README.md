# Restaurants Review System
A python script to collect the reviews of worst restaurants in Delhi using zomato's api and cluster them to figure why they were worst using K Means Machine Learning algorithm.

K-means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.

The problem is computationally difficult (NP-hard); however, there are efficient heuristic algorithms that are commonly employed and converge quickly to a local optimum. These are usually similar to the expectation-maximization algorithm for mixtures of Gaussian distributions via an iterative refinement approach employed by both algorithms. Additionally, they both use cluster centers to model the data; however, k-means clustering tends to find clusters of comparable spatial extent, while the expectation-maximization mechanism allows clusters to have different shapes.

In this project I hve called zomato api and then clustered all the worst restaurants of Delhi NCR on different parameters by Kmeans clustering .The results are shown below graphically.
![cluster_analysis](https://cloud.githubusercontent.com/assets/18600300/22627306/bee692a8-ebe6-11e6-9ee0-68a102d56c9a.png)
