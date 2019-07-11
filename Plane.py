from pyntcloud import PyntCloud
#import pyntcloud as pc
import numpy as np
from skimage import io
from scipy import stats
from skimage.feature import greycomatrix,greycoprops
import time
import pandas as pd
import math
from pyntcloud.ransac import models
from pyntcloud.ransac import fitters
from pyntcloud.scalar_fields import xyz
from pyntcloud.ransac.samplers import RandomRansacSampler
from pyntcloud.geometry.models.plane import Plane
from pyntcloud.ransac.models import RansacModel
from pyntcloud.scalar_fields import ALL_SF

'''From Line 21 to Line 97 , This part is inherited from the pyntcloud library and is modified for the
   our plane fitting task which required the equation of the Plane on the point cloud. '''

def equation(self):  #
    return self.get_equation()  #

models.RansacPlane.equation = equation 

def single_fit(points, model, sampler=RandomRansacSampler,
               model_kwargs={},
               sampler_kwargs={},
               max_iterations=100,
               return_model=False,
               n_inliers_to_stop=None):

    model = model(**model_kwargs)
    sampler = sampler(points, model.k, **sampler_kwargs)

    n_best_inliers = 0
    eq = [] #
    if n_inliers_to_stop is None:
        n_inliers_to_stop = len(points)

    for i in range(max_iterations):

        k_points = sampler.get_sample()

        if not model.are_valid(k_points):
            print(k_points)
            continue

        model.fit(k_points)

        all_distances = model.get_distances(points)

        inliers = all_distances <= model.max_dist

        n_inliers = np.sum(inliers)

        if n_inliers > n_best_inliers:
            n_best_inliers = n_inliers
            best_inliers = inliers
            eq = list(model.equation()) #

            if n_best_inliers > n_inliers_to_stop:
                break

    if return_model:
        model.least_squares_fit(points[best_inliers])
        return best_inliers, model, eq #

    else:
        return best_inliers, eq #

fitters.single_fit = single_fit

def compute(self):
        inliers, eq = single_fit(self.points, self.model, self.sampler,
                             model_kwargs=self.model_kwargs,
                             max_iterations=self.max_iterations,
                             n_inliers_to_stop=self.n_inliers_to_stop)
        self.to_be_added[self.name] = inliers.astype(np.uint8)
        return eq  #

xyz.PlaneFit.compute = compute

def add_scalar_field(self, name, **kwargs):
    
    if name in ALL_SF:
        scalar_field = ALL_SF[name](pyntcloud=self, **kwargs)
        scalar_field.extract_info()
        eq = scalar_field.compute()
        scalar_fields_added = scalar_field.get_and_set()

    else:
        raise ValueError("Unsupported scalar field. Check docstring")

    return scalar_fields_added, eq

PyntCloud.add_scalar_field = add_scalar_field

#Loading the point cloud as a Pandas DataFrame. Give Path in line 101 accordingly.

data = pd.read_csv('cloud(rgb).txt',sep = " ")
data.columns = ['x','y','z','red','green','blue','1','2','3','4','5']
#data = data.drop(['1','2','3','4','5'],axis = 1)
cloud = PyntCloud(data)

print(cloud)

#Cloud.points is a Pandas Dataframe.

#print(cloud.points.columns)

# Calculating the best fit plane on the point cloud data and getting the equation of the plane in return 
_,eq = cloud.add_scalar_field(name = 'plane_fit',max_dist = 5*1e-2)
print(eq)

a = eq[0]
b = eq[1]
c = eq[2]
d = eq[3]
#dis = abs((a * x1 + b * y1 + c * z1 + d))  
e = (math.sqrt(a * a + b * b + c * c)) 

# Making a mask over the points where the inliers of the best fit plane are selected 
mask = cloud.points['is_plane'] == 1

np.savetxt(r'Plane_inliers.txt',cloud.points[mask], fmt = '%f')
#cloud.points[mask]

# Removing the inliers from the Point Cloud
cloud.points = cloud.points.drop(cloud.points[mask].index)

# Calculating the distance of all the  points from the Best Fit Plane 
def distance(x1,y1,z1):
    return (a * x1 + b * y1 + c * z1 + d) / e
cloud.points['distance'] = list(map(distance,cloud.points['x'],cloud.points['y'],cloud.points['z']))

np.savetxt(r'Plane_outliers_behind.txt',cloud.points[cloud.points['distance'] < 0],fmt = '%f')
