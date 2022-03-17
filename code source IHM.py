import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from tkinter.filedialog import *
from tkinter import ttk
import csv
import pandas as pd
import xlrd
from xlwt import Workbook
import os
import os.path
from scipy.fft import fft, fftfreq
import scipy.fftpack
from scipy import fftpack
import cmath
import matplotlib as mpl
from scipy.signal import hilbert, chirp, butter, lfilter
from PIL import Image, ImageTk



elec=['FP1','FP2','FZ','F3','F4','F8','T3','C3','CZ','C4','T4','T5','P3','PZ','P4','T6','O1','OZ','O2']
Title = ["Bande Alpha", "Bande Beta", "Bande Theta", "Bande Delta"]
frequency_bands = ['alpha','beta','delta','theta']
state = ['Active','Death','Control']

def affichageMain():
    '''Cette fonction permet de créer le menu principal de l'IHM.
    '''

    #supression du bouton de démarrage précédent
    try: boutonStart
    except NameError : x=1
    else : boutonStart.destroy(), labelCassiopee.destroy(), labelTitle.destroy()

    fenetre.geometry("1000x700")
    bg = ImageTk.PhotoImage(file = directoryRepertoireCourant+"/etoile.jpg")
    canvas1 = Canvas(fenetre)
    canvas1.pack(fill = "both", expand = True)
    canvas1.create_image( 0, 0, image = bg, anchor = "nw")



    label = Label(fenetre, text="IHM EEG")
    label.pack()
    global directoryPatientEtat
    directoryPatientEtat = askdirectory(title="Choisir un patient")
    global patient
    global etat
    if len(directoryPatientEtat[len(path+"/"):].split("/"))==4:
        global band
        band=directoryPatientEtat[len(path+"/"):].split("/")[-3]
    patient,etat=directoryPatientEtat[len(path+"/"):].split("/")[-1],directoryPatientEtat[len(path+"/"):].split("/")[-2]

    #création du style d'onglets
    mygreen = "#d2ffd2"
    myred = "#dd0202"

    style = ttk.Style()

    style.theme_create( "yummy", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
        "TNotebook.Tab": {
            "configure": {"padding": [5, 1], "background": mygreen },
            "map":       {"background": [("selected", myred)],
                          "expand": [("selected", [1, 1, 1, 0])] } } } )

    style.theme_use("yummy")
    n = ttk.Notebook(fenetre)   # Création du système d'onglets
    n.pack()
    ongletMain = ttk.Frame(n)
    ongletMain.pack()
    n.add(ongletMain, text='Patient-Etat')

    modifierPatientEtat=Button(ongletMain, text='Modifier le patient',command=choisirPatientEtat)
    modifierPatientEtat.pack()

    ongletSpectrogram = ttk.Frame(n)
    ongletSpectrogram.pack()
    n.add(ongletSpectrogram, text='Spectogramme')
    carteTF=Button(ongletSpectrogram, text='Visualiser carte Temps-Fréquence',command=lambda:carteElectrode(displaySpectrogram))
    carteTF.pack()

    ongletFFT= ttk.Frame(n)
    ongletFFT.pack()
    n.add(ongletFFT, text='FFT')
    boutonFFT=Button(ongletFFT, text='Visualiser la FFT du signal',command=lambda:carteElectrode(FFT))
    boutonFFT.pack()

    ongletSynchro = ttk.Frame(n)
    ongletSynchro.pack()
    n.add(ongletSynchro, text='Synchronisation')
    global varBande
    varBande=IntVar()
    varBande.set(0)

    boutonMSC=Button(ongletSynchro, text='MSC',command=lambda:MSC_map(etat,patient))
    boutonMSC.grid(row=1,column=3)
    boutonPC=Button(ongletSynchro, text='PC',command=lambda:PC_map(etat,patient))
    boutonPC.grid(row=1,column=4)
    boutonPSC=Button(ongletSynchro, text='PSC',command=lambda:PSC_map(etat,patient))
    boutonPSC.grid(row=1,column=5)
    boutonPLI=Button(ongletSynchro, text='PLI',command=lambda:PLI_map(etat,patient))
    boutonPLI.grid(row=1,column=6)
    packBandeFrequenceSynchro(ongletSynchro,2)

    ongletSignaux = ttk.Frame(n)
    ongletSignaux.pack()
    n.add(ongletSignaux, text='Signaux bruts')
    signalElectrode=Button(ongletSignaux, text='Visualiser un signal',command=lambda:carteElectrode(displayCourbe))
    signalElectrode.grid(row=1,column=3)
    packBandeFrequenceRS(ongletSignaux,2)

    n.pack()
    onglet_canvas = canvas1.create_window(500, 100,window = n)

    fenetre.mainloop()

def packBandeFrequenceSynchro(onglet,n_row):
    '''Cette fonction actualise le choix de la bande de fréquence à sélectionner pour l'opération de calcul de matrices de cohérence et de synchronisation.
    '''
    global varBandeSynchro
    varBandeSynchro=IntVar()
    varBandeSynchro.set(0)
    choixFreq = ["Toutes","Alpha","Bêta","Delta","Theta"]
    choixBande=[]
    for i in range(5):
        choixBande.append(Checkbutton(onglet,text=choixFreq[i], onvalue=i, variable = varBandeSynchro))
        choixBande[i].grid(row=n_row,column=i+1)

def packBandeFrequenceRS(onglet,n_row):
    '''Cette fonction actualise le choix de la bande de fréquence à sélectionner pour la visualisation de signaux bruts.
    '''
    global varBandeRS
    varBandeRS=IntVar()
    varBandeRS.set(0)
    choixFreq = ["Toutes","Alpha","Bêta","Delta","Theta"]
    choixBande=[]
    for i in range(5):
        choixBande.append(Checkbutton(onglet,text=choixFreq[i], onvalue=i, variable = varBandeRS))
        choixBande[i].grid(row=n_row,column=i+1)

def carteElectrode(fonction):
    '''Cette fonction affiche la fenêtre permettant de sélectionner une électrode pour une opération particulière en paramètre.
    '''
    carte=Tk()
    carte.geometry("1100x900")
    carteLegende = Label(carte, text="Position des électrodes sur le cerveau")
    carteLegende.pack()

    #création des boutons électrodes
    boutonFP1=Button(carte,text='FP1',command=lambda:fonction("FP1"))
    boutonFP2=Button(carte,text='FP2',command=lambda:fonction("FP2"))
    boutonT3=Button(carte,text='T3',command=lambda:fonction("T3"))
    boutonC3=Button(carte,text='C3',command=lambda:fonction("C3"))
    boutonCZ=Button(carte,text='CZ',command=lambda:fonction("CZ"))
    boutonC4=Button(carte,text='C4',command=lambda:fonction("C4"))
    boutonT4=Button(carte,text='T4',command=lambda:fonction("T4"))
    boutonT5=Button(carte,text='T5',command=lambda:fonction("T5"))
    boutonT6=Button(carte,text='T6',command=lambda:fonction("T6"))
    boutonP3=Button(carte,text='P3',command=lambda:fonction("P3"))
    boutonP4=Button(carte,text='P4',command=lambda:fonction("P4"))
    boutonO1=Button(carte,text='O1',command=lambda:fonction("O1"))
    boutonOZ=Button(carte,text='OZ',command=lambda:fonction("OZ"))
    boutonO2=Button(carte,text='O2',command=lambda:fonction("O2"))

    #placement des boutons électrodes selon leur répartition géographique sur un vrai crâne humain.
    boutonFP1.place(x=450,y=75)
    boutonFP2.place(x=600,y=75)
    boutonT3.place(x=250,y=250)
    boutonC3.place(x=450,y=250)
    boutonCZ.place(x=525,y=250)
    boutonC4.place(x=600,y=250)
    boutonT4.place(x=800,y=250)
    boutonT5.place(x=350,y=375)
    boutonT6.place(x=700,y=375)
    boutonP3.place(x=400,y=275)
    boutonP4.place(x=650,y=275)
    boutonO1.place(x=450,y=450)
    boutonOZ.place(x=525,y=450)
    boutonO2.place(x=600,y=450)

    #exception instaurant une valeur aux abscisses/ordonnées minimales/maximales en cas de non-existence
    try:
        xMin
    except NameError:
        xMin= 0
    try:
        xMax
    except NameError:
        xMax= 20
    try:
        yMin
    except NameError:
        yMin= -20
    try:
        yMax
    except NameError:
        yMax= 20
    global sliderXmin
    global sliderXmax
    global sliderYmin
    global sliderYmax

    Label(carte, text="Réglages de la fenêtre").place(x=460,y=500)

    #création des curseurs
    sliderXmin = Scale(carte, from_=0, to=xMax,orient="horizontal",label="X min")
    sliderXmax = Scale(carte, from_=xMin, to=20,orient="horizontal",label="X max")
    sliderYmin = Scale(carte, from_=yMax, to=-20,orient="vertical",label="Y min")
    sliderYmax = Scale(carte, from_=20, to=yMin,orient="vertical",label="Y max")

    #ajustement des curseurs selon la fonction
    if fonction==FFT:
        sliderXmin.place(x= 400, y=550)
        sliderXmax.place(x= 600, y=550)
        sliderXmin.set(0)
        sliderXmax.set(20)


    if fonction==displayCourbe:
        sliderXmin.place(x= 300, y=550)
        sliderXmax.place(x= 400, y=550)
        sliderXmin.set(0)
        sliderXmax.set(20)
        sliderYmin.place(x= 600, y=550)
        sliderYmax.place(x= 710, y=550)
        sliderYmin.set(-20)
        sliderYmax.set(20)

    if fonction==displaySpectrogram:
        sliderYmin = Scale(carte, from_=25, to=0,orient="vertical",label="Y min")
        sliderYmax = Scale(carte, from_=25, to=0,orient="vertical",label="Y max")
        sliderXmin.place(x= 300, y=550)
        sliderXmax.place(x= 400, y=550)
        sliderXmin.set(0)
        sliderXmax.set(20)
        sliderYmin.place(x= 600, y=550)
        sliderYmax.place(x= 710, y=550)
        sliderYmin.set(0)
        sliderYmax.set(25)

def choisirPatientEtat():
    '''Cette fonction permet de sélectionner un patient, en demandant à l'utilisateur de choisir le bon dossier contenant les fichiers csv des électrodes. Ce dossier représente le patient.
    '''
    global directoryPatientEtat
    directoryPatientEtat = askdirectory(title="Choisir un patient")
    global patient
    global etat
    if len(directoryPatientEtat[len(path+"/"):].split("/"))==4:
        global band
        band=directoryPatientEtat[len(path+"/"):].split("/")[-3]
    patient,etat=directoryPatientEtat[len(path+"/"):].split("/")[-1],directoryPatientEtat[len(path+"/"):].split("/")[-2]


def getSubSampledData(electrode):
    global directoryPatientEtat
    global patient
    global etat
    path=directoryPatientEtat+"/"+str(patient)+"_"+str(electrode)+".csv"
    E = []
    cr=csv.reader(open(path))
    for row in cr:
        E.append(row)
    return E


def displaySpectrogram(electrode,fech=100):
    '''Cette fonction affiche la carte temps-fréquence d'une électrode en paramètre, d'un patient déjà sélectionné
    comprenant son état, ainsi que la bande de fréquence de filtrage si sélectionnée.
    '''
    global etat
    global patient
    E = []
    data = getSubSampledData(electrode)[0]
    for e in data:
        E.append(float(e))
    T = [k/fech for k in range(0,len(E))]
    plt.specgram(E,1000,fech)
    plt.title(str(electrode))
    #plt.ylim([0,25])
    if sliderXmax.get()<sliderXmin.get():
        plt.xlim(sliderXmax.get(),sliderXmin.get())
    else:
        plt.xlim(sliderXmin.get(),sliderXmax.get())
    if sliderYmax.get()<sliderYmin.get():
        plt.ylim(sliderYmax.get(),sliderYmin.get())
    else:
        plt.ylim(sliderYmin.get(),sliderYmax.get())
    plt.show()

def FFT(electrode,fech=100):
    '''Cette fonction affiche la FFT d'un signal brut d'une électrode en paramètre, d'un patient déjà sélectionné
    comprenant son état, ainsi que la bande de fréquence de filtrage.
    Cette fonction récupère le bon fichier csv de l'électrode, le lit et affiche sa représentation fréquentielle
    '''
    global patient
    global etat
    data = getSubSampledData(electrode)[0]
    T = [k/fech for k in range(0,len(data))]
    fig = plt.figure()
    sig_fft=fftpack.fft(data)
    power=np.abs(sig_fft)**2
    freq=fftpack.fftfreq(len(data),0.01)
    plt.plot(abs(freq),power)
    ax = plt.gca()
    ax.set_facecolor('k')
    if sliderXmax.get()<sliderXmin.get():
        plt.xlim(sliderXmax.get(),sliderXmin.get())
    else:
        plt.xlim(sliderXmin.get(),sliderXmax.get())
    plt.title(str(electrode)+' '+'Global'+' '+'Frequencies')
    plt.show()


def displayCourbe(electrode):
    '''Cette fonction affiche le signal brut d'une électrode en paramètre, d'un patient déjà sélectionné
    comprenant son état, ainsi que la bande de fréquence de filtrage.
    Cette fonction récupère le bon fichier csv de l'électrode, le lit et affiche sa représentation temporelle
    '''
    global directoryPatientEtat
    global patient
    global etat
    pathBande=directoryPatientEtat+"/"+str(patient)+"_"+str(electrode)+".csv"
    global varBandeRS
    if varBandeRS.get()>0:
        pathBande=path+"/"+"Données filtrées"+"/"+frequency_bands[varBandeRS.get()-1]+"/"+str(etat)+"/"+str(patient)+"/"+str(patient)+"_"+str(electrode)+".csv"
    else:
        pathBande=path+"/"+"Données sous-échantillonnées"+"/"+str(etat)+"/"+str(patient)+"/"+str(patient)+"_"+str(electrode)+".csv"
    global L
    L=[]
    cr=csv.reader(open(pathBande))
    for row in cr:
        for x in row:
            L.append(float(x))
    fig=plt.figure(figsize=(50,5))
    plt.subplot(111)
    plt.plot(np.linspace(0,20,len(L)),L)
    ax = plt.gca()
    ax.set_facecolor('k')
    if sliderXmax.get()<sliderXmin.get():
        plt.xlim(sliderXmax.get(),sliderXmin.get())
    else:
        plt.xlim(sliderXmin.get(),sliderXmax.get())
    if sliderYmax.get()<sliderYmin.get():
        plt.ylim(sliderYmax.get(),sliderYmin.get())
    else:
        plt.ylim(sliderYmin.get(),sliderYmax.get())
    if varBandeRS.get()==0:
        plt.title("électrode "+str(electrode)+", non filtré")
    else:
        plt.title("électrode "+str(electrode)+", bande "+str(frequency_bands[varBandeRS.get()-1]))
    plt.show()

#Lecture des données csv
def GetFilteredData(etat, N, band, electrode):
    os.chdir(path+'/'+'Données filtrées/'+band+'/'+etat+'/'+str(N))
    E = []
    cr = csv.reader(open(str(N)+"_"+str(electrode)+".csv","r"))
    for row in cr:
        E.append(row)
    return E

def GetSubSampledData(etat, N, electrode):
    os.chdir(path+'/'+'Données sous-échantillonées/'+etat+'/'+str(N))
    E = []
    cr = csv.reader(open(str(N)+"_"+str(electrode)+".csv","r"))
    for row in cr:
        E.append(row)
    return E


#Calcul des matrices de synchronisation sur les différentes bandes de fréquences pour un individu donné
def PSC(etat, N):
    global varBandeSynchro
    band=frequency_bands[varBandeSynchro.get()-1]
    os.chdir(path +'/'+"Données filtrées/"+ str(band)+"/" + str(etat) + "/")
    M = np.zeros((19,19))
    HilbertTransform = []
    for i in range(0,19):
        HilbertTransform.append(hilbert(GetFilteredData(etat,N,band,elec[i])[0]))
    for i in range(0,19):
        for j in range(0,19):
            phi_i = np.unwrap(np.angle(HilbertTransform[i]))
            T = []
            phi_j = np.unwrap(np.angle(HilbertTransform[j]))
            for k in range(0,len(phi_i)):
                T.append(np.exp(complex(0,phi_i[k]-phi_j[k])))
            avg = np.mean(T)
            gamma = np.abs(avg)
            M[i][j] = gamma
    return M

def PSC_map(etat, N):
    sortir=0
    global varBandeSynchro
    if varBandeSynchro.get()==0:
        fenetreError=Tk()
        label = Label(fenetreError, text="Veuillez sélectionner une bande de fréquence.")
        label.pack()
        fenetreError.mainloop()
        sortir=1
    if sortir==1:
        return 1
    band=frequency_bands[varBandeSynchro.get()-1]
    donnees = PSC(etat,N)
    plt.imshow(donnees, cmap = "jet")
    plt.colorbar()
    plt.title("PSC, "+str(band))
    plt.show()


#Calcul PLI
def PLI_deux_a_deux(sig1,sig2):
    hil1 = hilbert(sig1)
    hil2 = hilbert(sig2)
    N = len(hil1)
    pli = []
    for k in range(0,N):
        phi1 = np.angle(hil1[k])
        phi2 = np.angle(hil2[k])
        pli.append(np.sign(np.sin(phi1-phi2)))
    pli = np.abs(np.mean(pli))
    return pli

def PLI(etat, N):
    global varBandeSynchro
    band=frequency_bands[varBandeSynchro.get()-1]
    M = np.zeros((19,19))
    for i in range(0,19):
        for j in range(0,19):
            M[i][j] = PLI_deux_a_deux(GetFilteredData(etat,N,band,elec[i])[0],GetFilteredData(etat,N,band,elec[j])[0])
    return M

def PLI_map(etat, N):
    sortir=0
    global varBandeSynchro
    if varBandeSynchro.get()==0:
        fenetreError=Tk()
        label = Label(fenetreError, text="Veuillez sélectionner une bande de fréquence.")
        label.pack()
        fenetreError.mainloop()
        sortir=1
    if sortir==1:
        return 1
    band=frequency_bands[varBandeSynchro.get()-1]
    donnees = PLI(etat,N)
    plt.imshow(donnees,cmap='jet',vmin=0,vmax=1)
    plt.colorbar()
    plt.title("PLI, "+str(band))
    plt.show()


#Calcul coherence amplitude
def MSC_deux_a_deux(sig1, sig2):
    N = len(sig1)
    L = np.int(np.log2(N))-1
    M = np.int(2*N/L)-1
    D = np.int((N-L)/(M-1))
    x = np.zeros((L,M))
    y = np.zeros((L,M))
    for k in range(1,M+1):
        for i in range(1,L+1):
            x[i-1][k-1] = sig1[i + (k-1)*D]
            y[i-1][k-1] = sig2[i + (k-1)*D]
    fft_x = fftpack.fft(x)
    fft_y = fftpack.fft(y)
    MSC = np.zeros((L,1))
    for k in range(0,L):
        G_xx = np.mean(np.abs(fft_x[k])**2)
        G_yy = np.mean(np.abs(fft_y[k])**2)
        G_xy = np.mean(fft_x[k]*np.conjugate(fft_y[k]))
        MSC[k] = (np.abs(np.mean(G_xy))**2)/(np.mean(G_xx)*np.mean(G_yy))
    MSC = np.mean(MSC)
    return MSC

def MSC(etat, N):
    global varBandeSynchro
    band=frequency_bands[varBandeSynchro.get()-1]
    os.chdir(path +'/'+"Données filtrées/"+ str(band)+"/" + str(etat) + "/")
    M = np.zeros((19,19))
    for i in range(0,19):
        for j in range(0,19):
            M[i][j] = MSC_deux_a_deux(GetFilteredData(etat,N,band,elec[i])[0],GetFilteredData(etat,N,band,elec[j])[0])
    return M

def MSC_map(etat, N):
    sortir=0
    global varBandeSynchro
    if varBandeSynchro.get()==0:
        fenetreError=Tk()
        label = Label(fenetreError, text="Veuillez sélectionner une bande de fréquence.")
        label.pack()
        fenetreError.mainloop()
        sortir=1
    if sortir==1:
        return 1
    band=frequency_bands[varBandeSynchro.get()-1]
    donnees = MSC(etat,N)
    plt.imshow(donnees, cmap = "jet")
    plt.colorbar()
    plt.title("MSC, "+str(band))
    plt.show()


#Calcul de la coherence de phase
def PC_deux_a_deux(sig1,sig2):
    N = len(sig1)
    L = np.int(np.log2(N))-1
    M = np.int(2*N/L)-1
    D = np.int((N-L)/(M-1))
    x = np.zeros((L,M))
    y = np.zeros((L,M))
    for k in range(1,M+1):
        for i in range(1,L+1):
            x[i-1][k-1] = sig1[i + (k-1)*D]
            y[i-1][k-1] = sig2[i + (k-1)*D]
    fft_x = fftpack.fft(x)
    fft_y = fftpack.fft(y)
    PC = np.zeros((L,1))
    for k in range(0,L):
        phi = np.angle(np.mean(fft_x[k]*np.conjugate(fft_y[k])))
        PC[k] = phi
    PC = np.mean(PC)
    return PC

def PC(etat, N):
    global varBandeSynchro
    band=frequency_bands[varBandeSynchro.get()-1]
    M = np.zeros((19,19))
    for i in range(0,19):
        for j in range(0,19):
            M[i][j] = PC_deux_a_deux(GetFilteredData(etat,N,band,elec[i])[0],GetFilteredData(etat,N,band,elec[j])[0])
    return M

def PC_map(etat, N):
    sortir=0
    global varBandeSynchro
    if varBandeSynchro.get()==0:
        fenetreError=Tk()
        label = Label(fenetreError, text="Veuillez sélectionner une bande de fréquence.")
        label.pack()
        fenetreError.mainloop()
        sortir=1
    if sortir==1:
        return 1
    band=frequency_bands[varBandeSynchro.get()-1]
    donnees = PC(etat,N)
    plt.imshow(donnees,cmap='jet')
    plt.colorbar()
    plt.title("PC, "+str(band))
    plt.show()



global fenetre #création de la fenêtre
fenetre = Tk()
fenetre.geometry("450x300")
title=""
Lelectrode=["C3","C4","CZ","F3","F4","F7","F8","FP1","FP2","FZ","O1","O2","OZ","P3","P4","PZ","T3","T4","T5","T6"]
L=[]

global labelTitle #création du titre
labelTitle = Label(fenetre, text="Bienvenue dans l'IHM d'analyse de signaux EEG")
labelTitle.pack()

global directoryRepertoireCourant #variable du répertoire courant (localisation de l'IHM)
directoryRepertoireCourant = askdirectory(title="Choisir le répertoire courant (répertoire du dossier contenant l'IHM)")
path=directoryRepertoireCourant+"/Data"

image=Image.open(directoryRepertoireCourant+"/logoCassiopee.jpg",mode="r")
photo=ImageTk.PhotoImage(image)
global labelCassiopee
labelCassiopee=Label(fenetre,image=photo)
labelCassiopee.pack()

global boutonStart #bouton de lancement du programme
boutonStart=Button(text='Sélectionner un patient et son état', command=affichageMain)
boutonStart.pack(fill="x")
fenetre.mainloop()