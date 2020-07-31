import numpy as np
from retrodetect import detectcontact
from glob import glob
import pickle
import pandas as pd
import pickle
from libsvm.svmutil import svm_train,svm_predict,svm_load_model,svm_save_model
import matplotlib.pyplot as plt

def load_data(imgfiles,labelfile=None):
    """
    Load the images and the labelling (if any).
    
    labelfile, e.g. 'labels_photos_June08.csv'.
    imgfiles, e.g. a list of files
    
    returns a photo_list list of photoitems.
    """
    df = pd.read_csv(labelfile,names=['index','filename','x','y'])
    correctx = None
    correcty = None
    photo_list = []
    
    for imfilename in imgfiles:
        print("loading %s" % imfilename)
        #if (imfilename.find('39m')>=0): continue #don't train on the twoflash39m ones.
        
        photoitem = pickle.load(open(imfilename,'rb'))
        def getname(x):
            return x.split('/')[-1]
        dfs=df[df['filename'].apply(getname)==imfilename.split('/')[-1]]
        if len(dfs)>0: 
            correctx = dfs['x'].tolist()[0]
            correcty = dfs['y'].tolist()[0]

        if photoitem is not None:
            if photoitem['img'] is not None:
                photoitem['img'] = photoitem['img'].astype(np.float32)
            photoitem['correctx']=correctx
            photoitem['correcty']=correcty
            #print(correctx,correcty)
            #if len(photo_list)>0:
                #photo_list[-1]['correctx']=correctx
                #photo_list[-1]['correcty']=correcty
        if len(dfs)==0:
            correctx=None
            correcty=None
        photo_list.append(photoitem)
    print("%d photo items loaded." % len(photo_list))
    return photo_list

def drawreticule(x,y,c='w',alpha=0.5,angle=False):
    if angle:
        plt.plot([x-70,x-10],[y-70,y-10],c,alpha=alpha)
        plt.plot([x+70,x+10],[y+70,y+10],c,alpha=alpha)
        plt.plot([x-70,x-10],[y+70,y+10],c,alpha=alpha)
        plt.plot([x+70,x+10],[y-70,y-10],c,alpha=alpha)
    else:
        plt.hlines(y,x-70,x-10,c,alpha=alpha)
        plt.hlines(y,x+10,x+70,c,alpha=alpha)
        plt.vlines(x,y-70,y-10,c,alpha=alpha)
        plt.vlines(x,y+10,y+70,c,alpha=alpha)
        
def build_patches_lists(photo_list):
    correct = []
    incorrect = []
    for n in range(len(photo_list)):
        if not photo_list[n]['record']['endofset']: continue
        contact,found,searchimg = detectcontact(photo_list,n,delsize=100)
        if contact is None: continue
        if photo_list[n]['correctx'] is None: continue #we don't know where it is in the photo
        for c in contact:
            c['source']=photo_list[n-1]
            if ((photo_list[n]['correctx']-c['x'])**2 + (photo_list[n]['correcty']-c['y'])**2)<100:
                correct.append(c)
            else:
                incorrect.append(c)
    return correct, incorrect
                
def plot_photo_list(photo_list):
    for n in range(len(photo_list)):
        if not photo_list[n]['record']['endofset']: continue
        contact,found,searchimg = detectcontact(photo_list,n,delsize=100)

        if contact is None: continue
        if photo_list[n]['correctx'] is None: continue #we don't know where it is in the photo

        plt.figure(figsize=[25,20])
        img = photo_list[n-1]['img']
        plt.imshow(img)
        plt.clim([0,20])
        plt.colorbar()
        for c in contact:
            drawreticule(c['x'],c['y'])
            drawreticule(photo_list[n]['correctx'],photo_list[n]['correcty'],'y',1,angle=True)
            plt.title([photo_list[n]['index'],c['x'],c['y']])
            if c['prediction']<0:
                plt.gca().add_artist(plt.Circle((c['x'],c['y']), 50, color='w',fill=False))
                plt.gca().add_artist(plt.Circle((c['x'],c['y']), 52, color='k',fill=False))
                
def getstats(contactlist):
    res = []
    for i,c in enumerate(contactlist):
        outersurround = max(c['patch'][16,20],c['patch'][20,16],c['patch'][24,20],c['patch'][20,24],c['patch'][16,16],c['patch'][16,24],c['patch'][24,16],c['patch'][24,24])
        innersurround = max(c['patch'][18,20],c['patch'][20,18],c['patch'][22,20],c['patch'][20,22],c['patch'][18,18],c['patch'][18,22],c['patch'][22,18],c['patch'][22,22])
        centre = np.sum([c['patch'][20,20],c['patch'][20,21],c['patch'][20,19],c['patch'][19,20],c['patch'][21,20]])
        res.append([c['searchmax'],c['centremax'],c['mean'],outersurround,innersurround,centre])
    return np.array(res)
        
#### c['searchmax'],c['centremax'],c['mean'],outersurround,innersurround,centre
def generate_svm_object(correctstats,incorrectstats,dataset_file=None,svm_file=None):
    """
    Loads earlier data that's been submitted before, generates a new svm model, and optionally saves it.
    correctstats, incorrectstats = 
    """
    if len(correctstats)==0: return None
    if len(incorrectstats)==0: return None
    
    gamma = 3e-5
    C = 300
    
    if dataset_file is not None:
        try:
            oldcorrectstats,oldincorrectstats = pickle.load(open(dataset_file,'rb'))
            correctstats = np.unique(np.r_[oldcorrectstats,correctstats],axis=0)
            incorrectstats = np.unique(np.r_[oldincorrectstats,incorrectstats],axis=0)
        except FileNotFoundError:
            print("%s not found." % dataset_file)
    
    data = np.r_[incorrectstats,correctstats]
    target = np.r_[np.ones(len(incorrectstats)),np.zeros(len(correctstats))]
    if dataset_file is not None:
        pickle.dump((correctstats,incorrectstats),open(dataset_file,'wb'))
    m = svm_train(target,data,'-g %f -c %f -q' % (gamma,C))
    if svm_file is not None:
        svm_save_model(svm_file,m)
    return m


