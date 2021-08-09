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
webbrowser.open("file://" + os.path.realpath('index.html'),new=2)

pathtoimgs = args.imgpath #'/home/mike/Documents/Research/bee/photos2020/photos_June20'
label_data_file = args.labelfile #"labels_photos_June20.csv"
if 'port' in args: 
    port = args.port
else:
    port = 5000


def getimgfilename(number):
    fns = sorted(glob('%s/*.np'%(pathtoimgs)))
    return fns[number]

@app.route('/detect/<int:number>')
def detect(number):
    print("---------------------------------")
    photo_list = []
    print(number)
    for n in range(number-10,number+2):
        print(n)
        if n<0: continue
        fn = getimgfilename(n)
        print(fn)
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
    return jsonify({'contact':newcontact, 'found':found})
    
@app.route('/')
def hello_world():
    return 'root node of bee label API.'

@app.route('/filename/<int:number>')
def filename(number):
    return jsonify(getimgfilename(number))

@app.route('/configure/<string:path>')
def configure(path):
    global pathtoimgs
    pathtoimgs = path
    return "set new path %s" % path

@app.route('/savepos/<int:number>/<int:x1>/<int:y1>/<int:x2>/<int:y2>')
def savepos(number,x1,y1,x2,y2):
    print("==========================")
    fn = getimgfilename(number)
    print(number,fn,(x2+x1)/2,(y2+y1)/2)
    with open(label_data_file, "a") as labelfile:
        labelfile.write("%d,%s,%d,%d\n" % (number,fn,int((x2+x1)/2),int((y2+y1)/2)))


    return "done"
    
@app.route('/getimage/<int:number>/<int:x1>/<int:y1>/<int:x2>/<int:y2>')
def getimage(number,x1,y1,x2,y2):
    global pathtoimgs
    #print('%s/%04d'%(pathtoimgs,number))  
    print(x1,y1,x2,y2)
    #fns = sorted(glob('%s/*.np'%(pathtoimgs)))
    #if len(fns)==0:
    #    return "Image not found"
    fn = getimgfilename(number)
    print(fn)
    try:
        rawdata = np.load(fn,allow_pickle=True)
    except OSError:
        print("failed to load data")
        return jsonify({'index':-1,'photo':'failed','record':'failed'})
    if type(rawdata)==list:
        n, img, data = rawdata
    if type(rawdata)==dict:
        n = rawdata['index']
        img = rawdata['img']
        data = rawdata['record']
    if img is None:
        print("img = None")
        return jsonify({'index':-1,'photo':'failed','record':'failed'})
    print(img.shape)       
    steps = int((x2-x1)/500)
    if steps<1: steps = 1
    #img = (img.T[x1:x2:steps,y1:y2:steps]).T

    print(steps)
    img = (img.T[x1:x2,y1:y2]).T
    k = int(img.shape[0] / steps)
    l = int(img.shape[1] / steps)
    img = img[:k*steps,:l*steps].reshape(k,steps,l,steps).max(axis=(-1,-3))


    print(img.shape)
    #img[int(img.shape[0]/2),:] = 255
    #img[:,int(img.shape[1]/2)] = 255    
    return jsonify({'index':n,'photo':img.tolist(),'record':data})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=port)

