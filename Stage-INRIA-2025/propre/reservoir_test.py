import numpy as np
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.datasets import mackey_glass
from reservoirpy.nodes import Reservoir, Ridge, FORCE, ESN,Input
from reservoirpy.observables import mse

from numpy import *
import scipy as scp

rpy.verbosity(0)  # no need to be too verbose here
#rpy.set_seed(1234) # make everything reproducible!

# a cause de la taille de l'echantillon d'entrainement, le robot arrive au bout de 7500 itérations 
# dans la zone 2, le pattern ne saurait donc etre changé et la taille du setb non plus avant de savoir
# précisément dans quelle zone le robot est a la fin de l'entrainement 
# 


Data_set=np.load('data_set_bretenberg_noise.npy')
X=Data_set[:,:8]
Y=Data_set[:,11].reshape(-1,1)

"""X=np.load('input.npy')
Y=np.load('output.npy')
theta=np.load('positions.npy')
"""
training_len=7500
data_size=len(Y)
robot_model = {
                "N": 1000, 
                "sr": 0.58,            #0.5818465976198328, 
                "lr": 0.09,            #0.09209147803011875, 
                "input_scaling": 9.0,  # 8.978506821451717, 
                "seed": 784,           #26070 other seed
                "ridge": 0.13,         #0.13953840268837053, 
                "input_connectivity": 0.1,
                "connectivity": 0.1,
                "noise_rc": 0          #5.437647652528754e-05 
            }



reservoir = Reservoir(units=robot_model["N"],sr=robot_model["sr"], lr=robot_model["lr"] ,noise_rc=robot_model["noise_rc"],
                      input_connectivity=robot_model["input_connectivity"],rc_connectivity=robot_model["connectivity"], input_scaling=robot_model["input_scaling"],
                      input_bias=True, seed=robot_model["seed"])
readout = Ridge(ridge=robot_model["ridge"], output_dim=1,input_bias=True)
esn_model = reservoir >> readout
esn_model=esn_model.fit(X[:training_len],Y[:training_len],warmup=500)
print(reservoir.is_initialized, readout.is_initialized, readout.fitted)

esn_model.__dict__


def line_intersect(p1, p2, P3, P4):
    # calcule les coordonnées des points d'intersections entre deux segments. Ici entre le rayon du senseur et du mur

    # on récupère les coodonnées de tous les segments que formé par les senseurs
    p1 = np.atleast_2d(p1) 
    p2 = np.atleast_2d(p2)
    # on récupère les coordonnées des segments des murs du labyrinthe
    P3 = np.atleast_2d(P3)
    P4 = np.atleast_2d(P4)
    

    x1, y1 = p1[:,0], p1[:,1]
    x2, y2 = p2[:,0], p2[:,1]
    X3, Y3 = P3[:,0], P3[:,1]
    X4, Y4 = P4[:,0], P4[:,1]

    #colinéarité entre le vecteur senseur et le mur 
    D = (Y4-Y3)*(x2-x1) - (X4-X3)*(y2-y1)

    # Colinearity test
    C = (D != 0)
    UA = ((X4-X3)*(y1-Y3) - (Y4-Y3)*(x1-X3))
    UA = np.divide(UA, D, where=C)
    UB = ((x2-x1)*(y1-Y3) - (y2-y1)*(x1-X3))
    UB = np.divide(UB, D, where=C)

    # Test if intersections are inside each segment
    C = C * (UA > 0) * (UA < 1) * (UB > 0) * (UB < 1)
    
    X = np.where(C, x1 + UA*(x2-x1), np.inf)
    Y = np.where(C, y1 + UA*(y2-y1), np.inf)
    return np.stack([X,Y],axis=1)

def through_walls(position):
    x,y=position[0,0],position[0,1]
    wall_test=True
    if (x < 0 or x>300) or (y<0 or y>500) or (x>100 and x<200 and y>100 and y<200) or (x>100 and x<200 and y>300 and y<400): 
        wall_test=False
    return wall_test

pattern = [4,2,3,1,2,3]

def get_zones(position):
    if position[0,1] >350 : 
        return 1
    elif  position[0,1]<150:
        return 4
    elif position[0,0]<175 :
        return 2
    else : return 3


walls = np.array( [

            # Surrounding walls
            [ (  0,   0), (  0, 500)],
            [ (  0, 500), (300, 500)],
            [ (300, 500), (300,   0)],
            [ (300,   0), (  0,   0)],
            
            # Bottom hole
            [ (100, 100), (200, 100)],
            [ (200, 100), (200, 200)],
            [ (200, 200), (100, 200)],
            [ (100, 200), (100, 100)],

            # Top hole
            [ (100, 300), (200, 300)],
            [ (200, 300), (200, 400)],
            [ (200, 400), (100, 400)],
            [ (100, 400), (100, 300)]
        ] )

checkpoints= np.array( [

            # Surrounding walls
            [ (  0,   350), (  100, 350)],
            [ (  200,   350), (  300, 350)],
            [ (175, 300), (175,   200)],
            [ (  0,   150), (  100, 150)],
            [ (  200,   150), (  300, 150)],
        ] )

A = np.linspace(-np.pi/2, +np.pi/2, 8, endpoint=True)
sensors = {
    "angle" : A,
    "range" : 75*np.ones((8,1)),
    "value" : np.ones((8,1)) }
sensors["range"][3:5] *= 1.25
pred=20_000
print(sensors["range"])
orientation=np.array(Data_set[training_len,10])
position=np.array([Data_set[training_len,8],Data_set[training_len,9]]).reshape((1,2))
#position+= 2 * np.array([np.cos(float(orientation)),np.sin(float(orientation))]).reshape((1,2))
"""position
orientation="""
print(f'position:{position}')
print(f'orientation:{orientation}')


robot_size=10

naomi=np.zeros((pred,13))
zone=pattern[0]
idx=0
step_counter = 0
for n in range(pred):
    if through_walls(position)==True: 
        zonet1=get_zones(position)
        if zone == zonet1:
            zone=zone 
            step_counter += 1
            if step_counter >300:
                break
        else: 
            if zonet1==pattern[(idx+1)%6]: 
                idx+=1
                zone=zonet1
                step_counter=0
            else: break
        A = sensors["angle"] + orientation # angles de tous les senseurs
        T = np.stack([np.cos(A), np.sin(A)], axis=1) # gradient ? de tous les angles des senseurs
        P1 = position + robot_size*T # positions dans le plan de tous les senseurs du robot
        P2 = P1 + sensors["range"]*T # coordonées de la fin du segment formé a partir du senseur et jusq'uà sa distance de vision
        P3, P4 = walls[:,0], walls[:,1] # coordonnées de tous les murs du labyrinthe 
        for i, (p1, p2) in enumerate(zip(P1,P2)): # pour tous les segments robot-range 
            C = line_intersect(p1, p2, P3, P4) # calcul du point d'intersection entre la vision et le mur du labyrinthe
            index = np.argmin( ((C - p1)**2).sum(axis=1)) # somme des distances au carré du robot aux intersections, on obtient le nom du senseur de plus petit distance au mur 
            p = C[index] # on récupère la plus petite distance au mur
            if p[0] < np.inf: # si la distance est finie alors:
                sensors["value"][i] = np.sqrt(((p1-p)**2).sum())#+np.random.normal(loc=0,scale=1)
                sensors["value"][i] /= sensors["range"][i]
            else:
                sensors["value"][i] = 1
        naomi[n,:8]=sensors["value"].reshape((8,))
        dv=float(esn_model.run(naomi[n,:8])[0, 0])
        if abs(dv) > 0.5:
            orientation += 0.01*dv
            #print(orientation)
        else: dv=0
        naomi[n,9]=dv
        position += 2 * np.array([np.cos(float(orientation)),np.sin(float(orientation))]).reshape((1,2))
        naomi[n,10],naomi[n,11]=position[0,0],position[0,1]
        naomi[n,12]=orientation
    else : 
        break
print("\n")
print("-"*10,"fin de parcours","-"*10)
print(f"nombre de pas effectué avant erreur : {n}")
print(f"zone de l'erreur : {zonet1} , au pas {n}, nombre de tours: {idx//6},nombre de zones correctement passées: {idx}")
print("\n")
fig, ax = plt.subplots(figsize=(8, 8))

# Tracé des murs
for wall in walls:
    (x1, y1), (x2, y2) = wall
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
"""for check in checkpoints:
    (x1, y1), (x2, y2) = check
    ax.plot([x1, x2], [y1, y2], '--', linewidth=2, color='purple', alpha=0.1)"""
# Tracé de la trajectoire du robot
ax.plot(Data_set[:training_len,8],Data_set[:training_len,9],label='Trajectoire initiale', alpha=0.5)
ax.plot(naomi[:n,10], naomi[:n,11], 'r-', label='Trajectoire reservoir')
ax.plot(Data_set[training_len,8],Data_set[training_len,9], 'x', label='point de depart reservoir', markersize=10)
# Options d'affich
ax.set_xlim(0, 300)
ax.set_ylim(0, 500)
ax.set_aspect('equal')
#ax.set_title("Étape 2: trajectoire d'entrainement")
ax.set_title(f"nombre de zones parcourues : {idx} et {idx//6} tours en {n} pas")
ax.legend()
plt.grid(True)
plt.show()