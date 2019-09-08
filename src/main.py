import numpy as np 
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0) #data holds all the fetched and formated data with a minimum rating of 4.0

