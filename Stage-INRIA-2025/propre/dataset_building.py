import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(123)

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

class Maze:
    """
    A simple 8-maze made of straight walls (line segments)
    """

    def __init__(self):
        self.walls = np.array( [

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
            [ (100, 400), (100, 300)],

            # Moving walls (invisibles) to constraing bot path
            [ (  0, 250), (100, 200)],
            [ (200, 300), (300, 250)] 
        ] )

class Bot:
    
    def __init__(self):
        self.size = 10
        self.position = 150,250
        self.orientation = 0
        self.n_sensors = 8
        self.delta_theta=0
        self.tour=0
        A = np.linspace(-np.pi/2, +np.pi/2, self.n_sensors+2, endpoint=True)[1:-1]
        self.sensors = {
            "angle" : A,
            "range" : 75*np.ones((self.n_sensors,1)),
            "value" : np.ones((self.n_sensors,1)) }
        self.sensors["range"][3:5] *= 1.25
    
    def update(self, maze,invisible_walls):
        # Sensors
        A = self.sensors["angle"] + self.orientation # angles de tous les senseurs
        #print(A)
        T = np.stack([np.cos(A), np.sin(A)], axis=1) # gradient ? de tous les angles des senseurs
        P1 = self.position + self.size*T # positions dans le plan de tous les senseurs du robot
        P2 = P1 + self.sensors["range"]*T # coordonées de la fin du segment formé a partir du senseur et jusq'uà sa distance de vision
        if invisible_walls:
            P3, P4 = maze.walls[:,0], maze.walls[:,1] # coordonnées de tous les murs du labyrinthe avce murs invisibles
        else:
             P3, P4 = maze.walls[:12,0], maze.walls[:12,1]
        for i, (p1, p2) in enumerate(zip(P1,P2)): # pour tous les segments robot-range 
            C = line_intersect(p1, p2, P3, P4) # calcul du point d'intersection entre la vision et le mur du labyrinthe
            index = np.argmin( ((C - p1)**2).sum(axis=1)) # somme des distances au carré du robot aux intersections, on obtient le nom du senseur de plus petit distance au mur 
            p = C[index] # on récupère la plus petite distance au mur
            if p[0] < np.inf: # si la distance est finie alors:
                self.sensors["value"][i] = np.sqrt(((p1-p)**2).sum())#np.random.normal(0,1)
                self.sensors["value"][i] /= self.sensors["range"][i]
                self.sensors["value"][i]+=np.random.normal(loc=0,scale=0.5)
            else:
                self.sensors["value"][i] = 1

def update(invisible_walls=True):
    bot.delta_theta = (bot.sensors["value"].ravel() * [-4,-3,-2,-1,1,2,3,4]).sum()
    #print(bot.sensors["value"])
    if abs(bot.delta_theta) > 0.5:
        bot.orientation += 0.01 * bot.delta_theta
    bot.position += 2 * np.array([np.cos(bot.orientation),
                                  np.sin(bot.orientation)])
    bot.update(maze,invisible_walls)

    # Moving walls
    if invisible_walls:
        if bot.position[1] < 100 :
            maze.walls[12:] = [[(0, 250), (100, 300)], [(200, 200), (300, 250)]]
        elif bot.position[1] > 400:
            maze.walls[12:] = [[(0, 250), (100, 200)], [(200, 300), (300, 250)]]
    return 

def capteur(orientation,position,invisibles_walls=False):
    sensors_value=np.zeros((8,))
    A = bot.sensors["angle"] + orientation # angles de tous les senseurs
    #print(A)
    T = np.stack([np.cos(A), np.sin(A)], axis=1) # gradient ? de tous les angles des senseurs
    P1 = position + bot.size*T # positions dans le plan de tous les senseurs du robot
    P2 = P1 + bot.sensors["range"]*T # coordonées de la fin du segment formé a partir du senseur et jusq'uà sa distance de vision
    if invisible_walls:
        P3, P4 = maze.walls[:,0], maze.walls[:,1] # coordonnées de tous les murs du labyrinthe avce murs invisibles
    else:
            P3, P4 = maze.walls[:12,0], maze.walls[:12,1]
    for i, (p1, p2) in enumerate(zip(P1,P2)): # pour tous les segments robot-range 
        C = line_intersect(p1, p2, P3, P4) # calcul du point d'intersection entre la vision et le mur du labyrinthe
        index = np.argmin( ((C - p1)**2).sum(axis=1)) # somme des distances au carré du robot aux intersections, on obtient le nom du senseur de plus petit distance au mur 
        p = C[index] # on récupère la plus petite distance au mur
        if p[0] < np.inf: # si la distance est finie alors:
            sensors_value[i] = np.sqrt(((p1-p)**2).sum())#+np.random.normal(0,1)
            sensors_value[i] /= bot.sensors["range"][i]
            #self.sensors["value"][i]+=np.random.normal(loc=0,scale=0.5)
        else:
            sensors_value[i] = 1
    return sensors_value

pattern = [2,3,4,2,3,1]

def get_zones(position):
    if position[1] >350 : 
        return 1
    elif  position[1]<150:
        return 4
    elif position[0]<175 :
        return 2
    else : return 3
    

def through_walls(position):
    x,y = position
    wall_test=True
    if (x < 0 or x>300) or (y<0 or y>500) or (x>100 and x<200 and y>100 and y<200) or (x>100 and x<200 and y>300 and y<400): 
        wall_test=False
    return wall_test




#on construit le data set pour entrainer le modèle, 10_000 données recoupant les valeurs des senseurs, le theta, la position

# initialisation des variables
maze = Maze()
bot = Bot()
bot.position = 150, 250
orientation = 0
bot.tour=0
steps= 10_000
#  création du tableau 
set=np.zeros((steps,3))

for i in range(steps):
    sensors_data=bot.sensors["value"].ravel()
    #print(sensors_data)
    delta_theta= [bot.delta_theta]
    position= bot.position #+(np.random.normal(loc=0,scale=1,size=2)*0.7)
    if 149<position[0]<151 and 200<position[1]<300 and i>2:
        bot.tour+=1
        tour=[bot.tour]
        if bot.tour==2:
            completed_tour=i
            print(completed_tour)
    else: tour=[0]
    set[i]=np.concatenate((position,delta_theta))
    update()

#----------------------------------------------------------------------------

invisible_walls=False

data_set=np.zeros((steps,12))

"""
0:8 : capteurs
8 : coordonnées x
9: coordonnées y 
10: orientation générale du robot
11: delta angle bretenberg

"""
i=0
zone=get_zones(set[0,0:2])
print(set[0,0:2])
print(f"zone de départ du robot: {zone}")
for n in range(steps):
    
    data_set[n,8:10]=set[n,0:2] # reutilisation des positions du robot
    zonet1=get_zones(set[n,0:2])
    if zone == zonet1:
        zone=zone 
    else: 
        if zonet1==pattern[(i+1)%6]: 
            i+=1
            print(f"zone : {zonet1} , au pas {n}, nombre de tours: {i//6}, {i%6}")
            zone=zonet1
        else: break
    data_set[n,:8]=capteur(orientation,set[n,0:2])
    data_set[n,11]=set[n,2]
    data_set[n,10]=orientation
    orientation+=0.01*set[n,2]
print(i)
print(i//6)


np.save('data_set_bretenberg_noise.npy',data_set)

n_sensors=12
label=['capteur','capteur','capteur','capteur','capteur','capteur','capteur','capteur',
'8 : coordonnées x',
'9: coordonnées y', 
'10: orientation générale du robot',
'11: delta angle bretenberg'
]


plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(8, 1, i+1)
    plt.plot(data_set[:200,i], alpha=0.6)
    plt.ylabel(f"C{i}", loc='center')
    plt.xlim((0,200))
plt.xlabel(f"Étape 2: calcul des capteurs")
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))

# Tracé des murs
for wall in maze.walls[:12]:
    (x1, y1), (x2, y2) = wall
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

# Couleurs différentes pour chaque zone
colors = ['red', 'green', 'blue', 'orange']

zones= np.array([ # checkpoints zones positions
            [ (  0,   500), (  300, 350)],
            [ ( 0,   350), (  175, 150)],
            [ (  175,   350), (  300, 150)],
            [ (  0,   150), (  300, 0)],
            ])

# Tracé des zones (checkpoints)
for idx, ((x1, y1), (x2, y2)) in enumerate(zones):
    width = x2 - x1
    height = y2 - y1
    rect = Rectangle(
        (x1, y1),
        width,
        height,
        linewidth=0,
        edgecolor=colors[idx % len(colors)],
        facecolor=colors[idx % len(colors)],
        alpha=0.2,
        label=f"Zone {idx + 1}"
    )
    ax.add_patch(rect)

#ax.plot([200, 300], [300, 250], '--', linewidth=2)
#ax.plot([0, 100], [250, 300], '--', linewidth=2)
#ax.plot([200, 300], [200, 250], '--', linewidth=2)

# Tracé de la trajectoire du robot
ax.plot(set[:720,0],set[:720,1],label='Trajectoire initiale')
#ax.plot(set[750,0],set[750,1],'x',label='fin de course')
# Options d'affich
ax.set_xlim(0, 300)
ax.set_ylim(0, 500)
ax.set_aspect('equal')
ax.set_title('Zones de controle')
#ax.legend()
plt.grid(True)
plt.show()


