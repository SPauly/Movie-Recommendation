import numpy as np 
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0) #data holds all the fetched and formated data with a minimum rating of 4.0

print(repr(data['train'])) #print out the size of training and test data
print(repr(data['test']))

#create model
model = LightFM(loss='warp') # choose a loss = weighted approximate-rank pairwise

#train the model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    #number of users and items
    n_users, n_items = data['train'].shape

    #go through every user and assigne a recommendation
    for user_id in user_ids:
        #store the movies they already like
        liked_movies = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #our prediction
        score = model.predict(user_id,np.arange(n_items))

        #top items
        top_items = data['item_labels'][np.argsort(-score)]

        #print the results
        print("User %s" % user_id)
        print("Liked movies: ")

        for x in liked_movies[:3]:
            print("     %s" % x)

        print("Recommended: ")

        for x in top_items[:3]:
            print("     %s" % x)
    
sample_recommendation(model, data,[9,3,93])

