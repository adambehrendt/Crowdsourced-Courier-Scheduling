# Crowdsourced-Courier-Scheduling

## Description
This is the official data repository for the paper entitled "A Prescriptive Machine Learning Method for Courier Scheduling on Crowdsourced Delivery Platforms" submitted to Transportation Science [(Optimization Online Link)](http://www.optimization-online.org/DB_HTML/2021/10/8661.html). In this repository we include the datasets that we used in the paper that has been generated from real world crowdsourced delivery demand data according to our bootstrap sampling procedure, and hence is anonymized.

## Data
The data is stored in ```data_homogenous.csv``` and ```data_inhomogeneous.csv``` for homogeneous and inhomogeneous Poisson arrival processes, respectively. Each row of the ```*.csv``` files contains information describing the distributions of demand and ad-hoc courier arrivals for an operational period. That is, each row is to be used to generate multiple realizations of sets of orders. For a given row, the first 48 entries are what we refer to as the "pickup histogram" where each entry represents a 15 minute block of time and the probability that any individual order will have a pickup time in that time block (as such, the sum of the values is 1). The folling two entries are the mean and standard deviation of the distance between the O-D pairs of orders, following a normal distribution truncated at 0. Similarly, the next two entries are the mean and standard deviation of the total number of orders expected to arrive dynamically over the time horizon, following a normal distribution truncated at 0. The following entry is the final entry which describes demand and is the exact number of orders known at the beginning of the time horizon (i.e. the number of static orders). 

As for ad-hoc courier arrivals, we consider both homogeneous and inhomogeneous Poisson arrival processes and hence have a single additional feature representing the average arrival rate per period in the homogenous case and a feature vector of size 26 for the inhomogenous case, representing the arrival rate for each individual period.

For more detailed information on the structure of this data, how it was generated and our assumptions, please refer to the "Data Description, Processing and Assumptions" section of our paper.

## Sampling the Data