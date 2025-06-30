import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass

import reservoirpy as rpy
from reservoirpy.observables import nrmse, rsquare
import json
from reservoirpy.hyper import plot_hyperopt_report

rpy.verbosity(0)

Data_set=np.load('data_set bretenberg.npy')
X=Data_set[:,:8]
Y=Data_set[:,11].reshape(-1,1)
# Objective functions accepted by ReservoirPy must respect some conventions:
#  - dataset and config arguments are mandatory, like the empty '*' expression.
#  - all parameters that will be used during the search must be placed after the *.
#  - the function must return a dict with at least a 'loss' key containing the result
# of the loss function. You can add any additional metrics or information with other
# keys in the dict. See hyperopt documentation for more informations.

#┗ ┛┏ ┓━ ━ ┇┅┃

#┏━━━━━━━━━━┓
#┃     1    ┃
#┃   ┏━━┓   ┃
#┃┅┅┅┃  ┃┅┅┅┃
#┃   ┗━━┛   ┃
#┃ 2   ┇  3 ┃
#┃   ┏━━┓   ┃ 
#┃┅┅┅┃  ┃┅┅┅┃
#┃   ┗━━┛   ┃
#┃    X 4   ┃ X - bot's position at the end of the training
#┗━━━━━━━━━━┛

# the bot ends its "training course" in the zone 4 and come from 3
# so next it has to pass by zone 2

# functions used for the code 

pattern = [2,3,1,2,3,4]


def get_zones(position):
    if position[0,1] >350 : 
        return 1
    elif  position[0,1]<150:
        return 4
    elif position[0,0]<175 :
        return 2
    else : return 3
    

def through_walls(position):
    x,y=position[0,0],position[0,1]
    wall_test=True
    if (x < 0 or x>300) or (y<0 or y>500) or (x>100 and x<200 and y>100 and y<200) or (x>100 and x<200 and y>300 and y<400): 
        wall_test=False
    return wall_test

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

robot_size=10


def objective(dataset, config, *, input_scaling, N, sr, lr, ridge, seed, connectivity, input_connectivity,noise_rc):
    # This step may vary depending on what you put inside 'dataset'
    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.

    tours = []; zones = []; steps=[]
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(
            units=N,
            sr=sr,
            lr=lr,
            input_scaling=input_scaling,
            rc_connectivity=connectivity,
            input_connectivity= input_connectivity,
            noise_rc=noise_rc,
            seed=seed,
            input_bias=True
        )

        readout = Ridge(ridge=ridge, output_dim=1,input_bias=True)

        model = reservoir >> readout

        # Train your model and test your model.
        model.fit(X[:7500], Y[:7500], warmup=500) 

        # initialisation des paramtres du bot

        sensors = {
            "angle" : np.linspace(-np.pi/2, +np.pi/2, 8, endpoint=True), # sensors orientation
            "range" : 75*np.ones((8,1)), #length of each sensor
            "value" : np.ones((8,1)) } # initialisation of the sensors 
        sensors["range"][3:5] *= 1.25 # Both of front sensors' lengths are extended
       
        orientation=np.array(Data_set[7500,10]) # initialisation of bot's orientation
        position=np.array([Data_set[7500,8],Data_set[7500,9]]).reshape((1,2)) # bot's position
        
        pred=2500 # number of predictions for the test
        naomi=np.zeros((pred,13))
        
        zone=4 #zone at the end of the training 
        idx=0 # number of zones successfully passed by at the begining of the test
        
        step_counter = 0 # number of steps
        for n in range(pred):
            if through_walls(position)==True: 
                zonet1=get_zones(position)
                if zone == zonet1: # if the bot is still in its zone
                    step_counter += 1 
                    if step_counter >300: # if the number of step is too high
                        break    
                else: 
                    if zonet1==pattern[(idx+1)%6]: # if the bot's zones is the next in the pattern list
                        idx+=1
                        zone = zonet1
                        step_counter = 0

                    else: break #  if te bot is in the wrong zone 
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
                        sensors["value"][i] /= sensors["range"][i] #normalisation
                    else:
                        sensors["value"][i] = 1
                naomi[n,:8]=sensors["value"].reshape((8,))
                dv=float(model.run(naomi[n,:8])[0, 0])
                if abs(dv) > 0.5:
                    orientation += 0.01*dv
                    #print(orientation)
                else: dv=0
                naomi[n,9]=dv
                position += 2 * np.array([np.cos(float(orientation)),np.sin(float(orientation))]).reshape((1,2))
                naomi[n,10],naomi[n,11]=position[0,0],position[0,1]
                naomi[n,12]=orientation
            else : # the bot hit a wall
                break

        # save the datas for each test of hyperparameters 
        steps.append(n)
        tours.append(idx//6)
        zones.append(idx)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': 1/(1+np.mean(zones)),
            'zones': 1+np.mean(zones),
            'nb tours': np.mean(tours),
            'steps': steps, 
            'variable_seed':seed}

hyperopt_config = {
    "exp": "R-L",    # the experimentation name
    "hp_max_evals": 50,              # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",            # the method used by hyperopt to chose those sets (see below)
    "seed": 123,                       # the random state seed, to ensure reproducibility
    "instances_per_trial": 1,         # how many random ESN will be tried with each sets of parameters
    "hp_space": {                                                       # what are the ranges of parameters explored
        "N": ["choice", 1000],                                          # the number of neurons is fixed to 1000
        "sr": ["loguniform", 0.1, 0.9],                                 # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "lr": ["loguniform", 1e-2, 1],                                  # idem with the leaking rate, from 1e-3 to 1
        "input_scaling": ["loguniform", 0.1,0.9],                       # the input scaling is fixed
        "seed": ["choice", 12, 13, 14, 15, 16, 17, 18, 19, 20],         # an other random seed for the ESN initialization
        "ridge": ["loguniform", 1e-6, 1e-4],                            # and so is the regularization parameter.
        "input_connectivity": ["choice", 0.1],
        "connectivity": ["choice", 0.1],
        "noise_rc": ["loguniform", 1e-3, 1e-2],
    }

}

# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

dataset=1
from reservoirpy.hyper import research
best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

fig = plot_hyperopt_report(hyperopt_config["exp"], ("lr", "sr"), metric="loss")
