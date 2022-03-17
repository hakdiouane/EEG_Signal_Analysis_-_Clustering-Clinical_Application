Guide d'utilisation de l'IHM de visualisation de signaux EEG

Auteurs : Hakim DIOUANE, Aissam EL FARISSI, Yazid SAIAD

Bienvenue dans l'interface homme-machine de visualisation de signaux électroencéphalogrammes de patients pour des applications cliniques.
Cette application représente le projet Cassiopée de l'équipe n°61 pour l'édition 2021 et delivré à l'encontre de Mme Nesma HOUMANI.

L'IHM prend en entrée une certaine base de données de patients rangés par état (sains, repos ou morts), où chaque patient est représenté par une série de signaux d'électrodes (filtrés sur des bandes fréquences ou non).
Ces signaux sont au format .csv. L'application fonctionne uniquement selon l'arborescence de la base de données par défaut, qui est jointe dans le dossier contenant l'application. Ainsi, une autre base de données patients peut être considérée si cette dernière respecte l'arborescence.

Au démarrage, l'application demande de sélectionner le répertoire courant où se situe l'application. Cela permet de charger les images nécessaires au visuel de l'application. Mais surtout les données qui doivent être dans le repertoire courant et ordonnées selon l'arborescence de la base de données par défaut.

Une fois le répertoire courant sélectionné, l'application se lance et il faut choisir un patient à analyser. Ce patient est représenté par un dossier contenant des signaux d'électrodes aux formats .csv.

Lorsque le patient est sélectionné, nous avons alors 5 onglets :
 

1) Changement de patient 

L'utilisateur à la possibilité de sélectionner un autre patient en sélectionnant un dossier contenant des signaux d'électrodes.


2) Spectrogramme 

Cet onglet permet de visualiser la carte temps-fréquence associée à un patient et une électrode. Pour cela, l'IHM demande à l'utilisateur de choisir une électrode pour y visualiser la carte temps-fréquence associée. 
Une fenêtre s'ouvre et en cliquant sur une électrode, avec la possibilité d'ajuster la fenêtre, on peut visualiser la carte temps-fréquence correspondante.


3) FFT

Il est également possible de visualiser les transformées de Fourier des signaux des électrodes, à calculer par la FFT (Fast Fourier Transform). Comme précédemment, l'utilisateur atterit sur une fenêtre où les électrodes sont représentés par des boutons disposés géographiquement selon leur réel emplacement sur le cerveau du patient.
Chaque bouton permet d'afficher la FFT du signal de l'électrode sélectionnée.


4) Synchronisation

L'IHM offre la possiblité de visualiser les mesures de cohérence (MSC, PC) et de synchronisation (PSC, PLI) d'un patient sur une bande de fréquence donnée. 
Il s'agit tout simplement de cocher une bande de fréquence parmi les 4 (Alpha, Bêta, Delta, Theta) et d'appuyer sur le bouton de la mesure que l'on souhaite visualiser. 
La sélection d'une bande de fréquence est obligatoire, au quel cas un message d'erreur est retourné si aucune bande n'est choisie.

N.B. : Alpha [8-12 Hz], Bêta [12-25 Hz], Theta [4-8 Hz], Delta [1-4 Hz].


5) Signaux bruts

Enfin, on peut visualiser les signaux des électrodes de manière temporelle. Pour cela, il suffit de sélectionner si besoin une bande de fréquence, puis une électrode et si besoin ajuster la fenêtre (zoom).



Toutes les mesures s'affichent sur des fenêtres matplotlib, où il est possible de zoomer (en plus de l'ajustement de la fenêtre avant le lancement d'une visualisation) et de sauvegarder. 
Il est possible d'afficher succesivement plusieurs mesures en gardant les fenêtres des visualisations précédentes ouvertes ou fermées.