# PRANA: A contact tracing algorithm using DBSCAN

## Introduction

I am proposing a digital contact tracing algorithm that relies only on GPS data. Since I am not fluent with Bluetooth specs and due to limited time, I started experimenting using geospatial analysis techniques. I have published a detailed Medium article available at https://medium.com/swlh/building-a-simple-contact-tracing-model-using-the-dbscan-algorithm-5ea796d7afdc. 

### What is DBSCAN?

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** in machine learning is a popular unsupervised learning method used to separate clusters of high density from clusters of low density. The DBSCAN algorithm groups together data points that are close to each other and mark the outliers data points as noise.

### Instructions

* To run PRANA algorithm in your local system, you will require Jupyter Notebook available at https://jupyter.org/install.
* You will also need to install the required pip packages. I've made this easier for you. Just run `pip install --user -r requirements.txt` to install all the pip dependencies.


The algorithm is explained in the Jupyter Notebook with comments. If you have any suggestions or queries, please raise an issue.

### Future enhancements

* If time provides, I will explore more ways to perform contact tracing using other machine learning algorithms and techniques such as spacio temporal analysis etc.

## License

I have chosen to license my work under GNU GPLv3 license. 
Dependencies and derivative work are applicable under respective licenses.

Copyright (c) 2020 Madhusudhan Kumble and PRANA authors
