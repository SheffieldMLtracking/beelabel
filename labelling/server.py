from flask import Flask, make_response, jsonify
import numpy as np
from flask_cors import CORS
#from flask_compress import Compress
app = Flask(__name__)
#Compress(app)
CORS(app)
from glob import glob
from retrodetect import getblockmaxedimage
import argparse
import webbrowser
import os

import retrodetect

parser = argparse.ArgumentParser(description='Provide simple interface to label bee images')
parser.add_argument('--imgpath',required=True,type=str,help='Path to images')
parser.add_argument('--labelfile',required=True,type=str,help='File to save labels to, e.g. "labels_photos_June20.csv"')
#parser.add_argument('--initial',required=False,type=int,help='Initial image index')
parser.add_argument('--port',required=False,type=int,help='Port')
args = parser.parse_args()


pathtoimgsdir = args.imgpath #'/home/mike/Documents/Research/bee/photos2020/photos_June20'

pathtoimgs = sorted(glob(pathtoimgsdir+'/*/'))
assert len(pathtoimgs)>0, "Failed to find any folders in the path"
print("Found the following camera folders:")
print(pathtoimgs)
webbrowser.open("file://" + os.path.realpath('index.html'),new=2)
label_data_file = args.labelfile #"labels_photos_June20.csv"
if 'port' in args: 
    port = args.port
else:
    port = 5000


def getimgfilelist(path):
    return sorted(glob('%s/*.np'%(path)))
    
def getimgfilename(cam,number):
    fns = getimgfilelist(pathtoimgs[cam])
    if number>=len(fns): return None
    return fns[number]

def gethash(obj):
    """
    Returns a 160 bit integer hash
    """
    return int(hashlib.sha1(obj).hexdigest(),16)
    

@app.route('/detectfromto/<int:cam>/<int:from_idx>/<int:to_idx>')
def detectall(cam,from_idx,to_idx):
    for i in range(from_idx,to_idx):
        detect(cam,i)

import pickle
import hashlib
@app.route('/detect/<int:cam>/<int:number>')
def detect(cam,number):
    path = pathtoimgs[cam]
    cachefile = 'detect_cache_%s_%d.pkl' % (gethash(path.encode("utf-8")),number)
    try:
        result = pickle.load(open(cachefile,'rb'))
        print("Cache hit %s" % cachefile)
        return result
        
    except FileNotFoundError:
        pass
    photo_list = []
    for n in range(number-10,number+2):
        if n<0: continue
        fn = getimgfilename(cam,n)
        try:
            photoitem = np.load(fn,allow_pickle=True) 
        except OSError:
            continue #skip this one if we can't access it
        if photoitem is not None:
            if photoitem['img'] is not None:
                photoitem['img'] = photoitem['img'].astype(np.float16)
        photo_list.append(photoitem)
    contact, found, _ = retrodetect.detectcontact(photo_list,len(photo_list)-1,Npatches=50,delsize=5,blocksize=3)
    newcontact = []
    if contact is not None:
        for c in contact:
            c['patch']=c['patch'].tolist() #makes it jsonable
            c['searchpatch']=c['searchpatch'].tolist() #makes it jsonable
            c['mean']=float(c['mean'])
            c['searchmax']=float(c['searchmax'])
            c['centremax']=float(c['centremax'])
            c['x']=int(c['x'])
            c['y']=int(c['y'])
            newcontact.append(c)
    result = jsonify({'contact':newcontact, 'found':found})
    pickle.dump(result,open(cachefile,'wb'))
    return result
    
@app.route('/')
def hello_world():
    return 'root node of bee label API.'

@app.route('/filename/<int:cam>/<int:number>')
def filename(cam,number):
    return jsonify(getimgfilename(cam,number))

@app.route('/configure/<string:path>')
def configure(path):
    global pathtoimgs
    pathtoimgs = path
    return "set new path %s" % path

@app.route('/savepos/<int:cam>/<int:number>/<int:x1>/<int:y1>/<int:x2>/<int:y2>')
def savepos(cam,number,x1,y1,x2,y2):
    fn = getimgfilename(cam,number)
    with open(label_data_file, "a") as labelfile:
        labelfile.write("%d,%s,%d,%d\n" % (number,fn,int((x2+x1)/2),int((y2+y1)/2)))


    return "done"
    
@app.route('/getimage/<int:cam>/<int:number>/<int:x1>/<int:y1>/<int:x2>/<int:y2>')
def getimage(cam,number,x1,y1,x2,y2):
    global pathtoimgs

    #fns = sorted(glob('%s/*.np'%(pathtoimgs)))
    #if len(fns)==0:
    #    return "Image not found"
    fn = getimgfilename(cam,number)

    try:
        rawdata = np.load(fn,allow_pickle=True)
    except OSError:

        return jsonify({'index':-1,'photo':'failed','record':'failed'})
    if type(rawdata)==list:
        n, img, data = rawdata
    if type(rawdata)==dict:
        n = rawdata['index']
        img = rawdata['img']
        data = rawdata['record']
    if img is None:

        return jsonify({'index':-1,'photo':'failed','record':'failed'})

    steps = int((x2-x1)/500)
    if steps<1: steps = 1
    #img = (img.T[x1:x2:steps,y1:y2:steps]).T


    img = (img.T[x1:x2,y1:y2]).T
    k = int(img.shape[0] / steps)
    l = int(img.shape[1] / steps)
    img = img[:k*steps,:l*steps].reshape(k,steps,l,steps).max(axis=(-1,-3))



    #img[int(img.shape[0]/2),:] = 255
    #img[:,int(img.shape[1]/2)] = 255    
    return jsonify({'index':n,'photo':img.tolist(),'record':data})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=port)

