from __future__ import print_function
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


#convert to pandas dataframes
feat_cols = ['Domain','Year','Haptic','VR','AR','2D','NA','Touchpad',
             'Vibration','Wind','Temperature','Audio','Gizmo','Tracking',
             'NA','Object','Eye','Hand','Head','Body','NA','Survey',
             'StateOfTheArt','Review','System','Simulation','Experiment',
             'Analysis']
df = pd.DataFrame(pd.read_csv('./papers.csv'),columns=feat_cols)

print('Size of the dataframe: {}'.format(df.shape))

references=['Hekler,Klasnja,Froehlich, and Buman(2013). Bas','Kehrer and Hauser(2013). Bas','Reda et al.(2013). Bas','Vines,Clarke,Wright,McCarthy, and Olivier(2013). Bas','Achibet,Marchal,Argelaguet, and Lécuyer(2014). Bas','Deng,Kirkby,Chang, and Zhang(2014). Bas','Diefenbach,Kolb, and Hassenzahl(2014). Bas','Freina and Ott(2015). Bas','Muhanna(2015). Bas','Olshannikova,Ometov,Koucheryavy, and Olsson(2015). Bas',
            'Anthes et al.(2016). Bas','Chavan(2016). Bas','Slater and Sanchez-Vives(2016). Bas','Pacchierottiet al.(2017). Bas','Rubio-Tamayo,Gertrudix Barrio, and García García(2017). Bas','Talasaz and Patel(2013). Med','Díaz,Gil, and Louredo(2014). Med','Esteban,Fernández,Conde, and García-Pẽnalvo(2014). Med','Jeon and Harders(2014). Med','Khanal et al.(2014). Med',
            'Fortmeier,Wilms,Mastmeyer,and Handels(2015). Med','Pan et al.(2015). Med','Ruffaldi,Brizzi,Filippeschi,and Avizzano(2015). Med','Ruthenbeck and Reynolds(2015). Med','D.Wang,Zhao,Li,Zhang,and Wang(2015). Med','Andaluz et al.(2016). Med','Escobar-Castillejos,Noguez,Neri,Magana,and Benes(2016). Med','Vaughan,Dubey,Wainwright,and Middleton(2016). Med','D.Wang,Li,Zhang,and Hou(2016). Med','Won et al.(2017). Med',
            'Rose,Nam,and Chen(2018). Med','Hamza-Lup,Bogdan,Popovici,and Costea(2019). Med','Grane and Bengtsson(2013). Tran','Altendorf et al.(2014). Tran','Kemeny(2014). Tran','Mars,Deroo,and Hoc(2014). Tran','Aslandere,Dreyer,and Pankratz(2015). Tran','Li and Zhou(2016). Tran','Marayong et al.(2017). Tran','Oberhauser and Dreyer(2017). Tran',
            'Valentino,Christian,and Joelianto(2017). Tran','Z.Wang,Zheng,Kaizuka,and Nakano(2018). Tran','Stamer,Michaels,and Tümler(2020). Tran','Kucukyilmaz,Sezgin,and Basdogan(2013). Ph','Donalek et al.(2014). Ph','S.-C.Kim and Kwon(2014). Ph','Kokubun,Ban,Narumi,Tanikawa,and Hirose(2014). Ph','Nakamura and Yamamoto(2014). Ph','Prachyabrued and Borst(2014). Ph','Z.Wang and Wang(2014). Ph',
            'Madan,Kucukyilmaz,Sezgin,and Basdogan(2015). Ph','Amirkhani and Nahvi(2016). Ph','Lindgren,Tscholl,Wang,and Johnson(2016). Ph','Magana et al.(2017). Ph','Shaikh et al.(2017). Ph','Yuksel et al.(2017). Ph','Edwards,Bielawski,Prada,and Cheok(2018). Ph','Neri et al.(2018). Ph','Walsh et al.(2018). Ph','Yuksel et al.(2019). Ph',
            'Groten,Feth,Klatzky,and Peer(2013). UX','Kober and Neuper(2013). UX','Okamoto,Nagano,and Yamada(2013). UX','Aras,Shen,and Noor(2014). UX','Cavrag,Larivière,Cretu,and Bouchard(2014). UX','Hamam,Saddik,and Alja’am(2014). UX','Odom,Zimmerman,and Forlizzi(2014). UX','Achibet,Girard,Talvas,Marchal,and Lécuyer(2015). UX','Bombari,Schmid Mast,Canadas,and Bachmann(2015). UX','Fittkau,Krause,and Hasselbring(2015). UX',
            'Moran,Gadepally,Hubbell,and Kepner(2015). UX','Ahmed et al.(2016). UX','Atienza,Blonna,Saludares,Casimiro,and Fuentes(2016). UX','Carvalheiro,Nóbrega,da Silva,and Rodrigues(2016). UX','Y.-S.Chen et al.(2016). UX','Matsumoto et al.(2016). UX','M.Kim,Jeon,and Kim(2017). UX','Kyriakou,Pan,and Chrysanthou(2017). UX','Lee,Kim,and Kim(2017). UX','Maereg,Nagar,Reid,and Secco(2017). UX',
            'Piumsomboon,Lee,Lindeman,and Billinghurst(2017). UX','Reski and Alissandrakis(2019). UX','C.-Y.Chen,Chang,and Huang(2014). Cult','Dima,Hurcombe,and Wright(2014). Cult','Gaugne,Gouranton,Dumont,Chauffaut,and Arnaldi(2014). Cult','Pietroni and Adami(2014). Cult','Papaefthymiou,Plelis,Mavromatis,and Papagiannakis(2015). Cult','Jung,tom Dieck,Lee,and Chung(2016). Cult','Kersten,Tschirschwitz,and Deggim(2017). Cult','Tsai,Shen,Lin,Liu,and Chiou(2017). Cult',
            'Younes et al.(2017). Cult','Barbieri,Bruno,and Muzzupappa(2018). Cult','Bekele,Pierdicca,Frontoni,Malinverni,and Gain(2018). Cult','Carrozzino,Colombo,Tecchia,Evangelista,and Bergamasco(2018). Cult','Perret,Kneschke,Vance,and Dumont(2013). Ind','Qiu,Fan,Wu,He,and Zhou(2013). Ind','Xia,Lopes,and Restivo(2013). Ind','Gonzalez-Badillo,Medellin-Castillo,Lim,Ritchie,and Garbaya(2014). Ind','Hamid,Aziz,and Azizi(2014). Ind','Vélaz,Arce,Gutiérrez,Lozano-Rodero,and Suescun(2014). Ind',
            'Abidi,Ahmad,Darmoul,and Al-Ahmari(2015). Ind','Choi,Jung,and Noh(2015). Ind','Gavish et al.(2015). Ind','Grajewski,Górski,Hamrol,and Zawadzki(2015). Ind','Radkowski,Herrema,and Oliver(2015). Ind','Al-Ahmari,Abidi,Ahmad,and Darmoul(2016). Ind','X.Wang,Ong,and Nee(2016). Ind','Xia (2016). Ind','Berg and Vance(2017). Ind','Ho,Wong,Chua,and Chui(2018). Ind',
            'Loch,Ziegler,and Vogel-Heuser(2018). Ind','Roldán,Crespo,Martín-Barrio,Peña-Tapia,and Barrientos(2019). Ind']

# Instantiate the clustering model and visualizer
dendrogram = sch.dendrogram(sch.linkage(df, method = 'ward'),
                            color_threshold=15, labels=references)

plt.title('Dendrogram')
plt.xlabel('Paper reference')
plt.ylabel('Euclidean distances')
plt.axhline(y=15,c='k')
plt.show()

