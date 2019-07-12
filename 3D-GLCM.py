from pyntcloud import PyntCloud
import numpy as np
from skimage import io
from scipy import stats
from skimage.feature import greycomatrix,greycoprops
import time
import pandas as pd

start = time.time()

#Loading the point cloud from ASCII file in a custom data structure.Change the 'cloud(rgb).txt' in line 14 to any point cloud that you want to load.

#cloud = PyntCloud.from_file("cloud.ply")
data = pd.read_csv('cloud(rgb).txt',sep = " ")
data.columns = ['x','y','z','red','green','blue','1','2','3','4','5']
data = data.drop(['1','2','3','4','5'],axis = 1)
cloud = PyntCloud(data)
#cloud.points = data

print(cloud)

#Cloud.points is a Pandas Dataframe.

print(cloud.points.columns)

#A grayscale column is formed for the GLCM calculations.

cloud.points['grayscale'] = (cloud.points['red']+cloud.points['green']+cloud.points['blue']) / 3 
cloud.points['grayscale'] =cloud.points['grayscale'].astype(int)

#Voxelization of the point cloud to grid according to the sizes of the axis.
#TODO : Understand and dry run the procedure being applied during the formation of this grid.
#The cause of bad results. The sizes must be optimized according to the data to form good approximation of the point cloud.

cloud.add_structure(name ='voxelgrid',size_x = 0.2,size_y=0.2,size_z=0.2)

keys = []
for key in cloud.structures:
    keys.append(key)

#Choosing the VoxelGrid from the structures of the Cloud object.
voxel = cloud.structures[keys[0]] 

#Forming the 3D Matrices from the voxelization values.
arr = np.zeros((voxel.x_y_z[0],voxel.x_y_z[1],voxel.x_y_z[2]))
arr1 = np.zeros((voxel.x_y_z[0],voxel.x_y_z[1],voxel.x_y_z[2]))

# for i,row in cloud.points.iterrows(): 
#     # print(i)
#     # print(row['grayscale'])
#     arr[voxel.voxel_x[i],voxel.voxel_y[i],voxel.voxel_z[i]] += row['grayscale']
#     arr1[voxel.voxel_x[i],voxel.voxel_y[i],voxel.voxel_z[i]] += 1

#Adding the grayscale values in the 3D Matrice.
for i in range(len(cloud.points.index)):
    arr[voxel.voxel_x[i],voxel.voxel_y[i],voxel.voxel_z[i]] += cloud.points.at[i,'grayscale']
    arr1[voxel.voxel_x[i],voxel.voxel_y[i],voxel.voxel_z[i]] += 1    

#Averaging the values.
#TODO : Use Map to fasten this nested loop.
for i in range(voxel.x_y_z[0]):
    for j in range(voxel.x_y_z[1]):
        for k in range(voxel.x_y_z[2]):
            if(arr1[i,j,k]):
                arr[i,j,k] =  arr[i,j,k]//arr1[i,j,k] 

#print(np.max(arr))

#GLCM Operations For 3D matrices.
def offset_3d(length, angle):
    """Return the offset in pixels for a given length and angle"""
    dv = length[0] * np.sign(-np.sin(angle[0])).astype(np.int32)
    dh = length[0] * np.sign(np.cos(angle[0])).astype(np.int32)
    dz = length[1] * np.sign(np.sin(angle[1])).astype(np.int32)
    return dv, dh, dz 

def crop_3d(img, center, win):
    """Return a cubic crop of img centered at center (side = 2*win + 1)"""
    row, col, height = center
    side = 2*win + 1
    first_row = row - win
    first_col = col - win
    first_height = height - win
    last_row = first_row + side    
    last_col = first_col + side
    last_height = first_height + side
    return img[first_row: last_row, first_col: last_col, first_height: last_height]

def cooc_maps_3d(img, center, win, d=[1,1], theta=[0,0], levels=256):
    """
    Return a set of co-occurrence maps for different d and theta in a square 
    crop centered at center (side = 2*w + 1)
    """
    shape = (2*win + 1, 2*win + 1, 2*win + 1, 1, 1)
    cooc = np.zeros(shape=shape, dtype=np.int32)
    row, col, height = center
    Ii = crop_3d(img, (row, col, height), win)
    dv, dh, dz = offset_3d(d, theta)
    Ij = crop_3d(img, center=(row + dv, col + dh, height + dz), win=win)
    cooc[:, :, :, 0, 0] = encode_cooccurrence(Ii, Ij, levels)
    return cooc    

def encode_cooccurrence(x, y, levels=256):
    """Return the code corresponding to co-occurrence of intensities x and y"""
    x = (x/256)*levels
    y = (y/256)*levels
    x = x//1
    y = y//1
    return x*levels + y

def decode_cooccurrence(code, levels=256):
    """Return the intensities x, y corresponding to code"""
    return code//levels, np.mod(code, levels)    

def compute_glcms_3d(cooccurrence_maps, levels=256):
    """Compute the cooccurrence frequencies of the cooccurrence maps"""
    Nr, Na = cooccurrence_maps.shape[3:]
    glcms = np.zeros(shape=(levels, levels, Nr, Na), dtype=np.float64)
    for r in range(Nr):
        for a in range(Na):
            table = stats.itemfreq(cooccurrence_maps[:, :, :, r, a])
            codes = table[:, 0]
            freqs = table[:, 1]/float(table[:, 1].sum())
            i, j = decode_cooccurrence(codes, levels=levels)
            #print(i)
            #print(j)
            glcms[i, j, r, a] = freqs
    return glcms   

def compute_props(glcms, props=('contrast',)):
    """Return a feature vector corresponding to a set of GLCM"""
    Nr, Na = glcms.shape[2:]
    features = np.zeros(shape=(Nr, Na, len(props)))
    for index, prop_name in enumerate(props):
        features[:, :, index] = greycoprops(glcms, prop_name)
    return features.ravel()


def haralick_features_3d(img, win, d, theta, levels, props):
    """Return a map of Haralick features (one feature vector per pixel)"""
    x,y,z = img.shape
    margin = win + max(d)
    arr = np.pad(img, margin, mode='reflect')
    n_features = len(props)
    feature_map = np.zeros(shape=(x, y, z, n_features), dtype=np.float64)
    # arr = (arr/256)*levels
    # arr = arr//1
    # arr = arr.astype(int)
    for m in range(x):
        for n in range(y):
            for o in range(z):
                # print(o)
                coocs = cooc_maps_3d(arr, (m + margin, n + margin, o + margin), win, d, theta, levels)
                glcms = compute_glcms_3d(coocs, levels)
                # glcms = greycomatrix(crop(arr,(m+margin,n+margin),win),d,theta,levels=levels,symmetric=True,normed=True)
                feature_map[m, n, o, :] = compute_props(glcms, props)
    return feature_map

#Set variables for 3D GLCM operations
mat = arr 
win = 10
dis = [0,1] 
theta = [0,np.pi/2]
levels = 16
props = ['contrast','homogeneity','ASM','energy','correlation']
feature_map_3d = haralick_features_3d(mat, win, dis, theta, levels, props)

# Results in different columns of the feature_map. Uncomment to use other features.

result = feature_map_3d[:,:,:,0]  #Contrast
# result = feature_map_3d[:,:,:,1]  #Homogeneity
# result = feature_map_3d[:,:,:,2]  #ASM
# result = feature_map_3d[:,:,:,3]  #Energy
# result = feature_map_3d[:,:,:,4]  #Correlation

# Normalizing the result
result = (( result - np.min(result) )/( np.max(result)-np.min(result) )) * 255
 
# Mapping the results to the Point Cloud
for i in range(len(cloud.points.index)):
    cloud.points.at[i,'red'] = result[voxel.voxel_x[i],voxel.voxel_y[i],voxel.voxel_z[i]]
    cloud.points.at[i,'green'] = result[voxel.voxel_x[i],voxel.voxel_y[i],voxel.voxel_z[i]]
    cloud.points.at[i,'blue'] = result[voxel.voxel_x[i],voxel.voxel_y[i],voxel.voxel_z[i]]    

cloud.points = cloud.points.drop('grayscale',1)

#cloud.to_file("Contrast_05.txt")
np.savetxt(r'Contrast_02.txt',cloud.points.values,fmt = '%f')

end = time.time()

print(end-start)
