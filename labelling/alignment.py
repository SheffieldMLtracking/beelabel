import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import time
import glob
import matplotlib.patches as patches
import pickle
from scipy import optimize
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import retrodetect
import hashlib





    
def grab_photos_in_timerange(path,starttime,endtime):
    """
    Finds all numpy files (with timestamps in) at 'path' location. Avoid
    trailing / in this parameter.
    between starttime and endtime. These two parameters need to be either
    time.struct_time objects, e.g. time.strptime('12','%H'), or
    HH:MM:SS strings.
    
    Returns a list of ordered filenames.
    
    Example:
        grab_photos_in_timerange('photos/system001','08:00:00','08:10:00')
    """
    if type(starttime)==str: starttime = time.strptime(starttime,'%H:%M:%S')
    if type(endtime)==str: endtime = time.strptime(endtime,'%H:%M:%S')
    output = []
    fs = sorted(glob.glob(path+'/*.np'))
    for f in fs:
        fnd = re.findall('[0-9]*:[0-9]*:[0-9]*',f)
        if len(fnd)!=1: continue
        t = time.strptime(fnd[0],'%H:%M:%S')
        if (t>=starttime) & (t<=endtime):
            output.append(f)
    return output

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def getcode(save,im,debug=False,shift=None,thresholds = [1.5,0.6]):
    """
    Reads the code from a post in the image, using the config in 'save'.
    save should be a dictionary containing:
    loc - a tuple of the location in the image
    scale - the width of the post
    angle - the tilt angle of the post
    
    returns the code or None if failed.
    """
    
    #we initially do a rough sampling down the length of the post, and later look at that signal
    
    #find the top of the post, and the step size in x/y to move down the post
    
    
    xstep = np.sin(np.deg2rad(save['angle']))*save['scale']
    ystep = np.cos(np.deg2rad(save['angle']))*save['scale']
    x,y = save['loc']
    if shift is not None: y = y + shift
    startx=x+save['scale']*1.5-xstep*4
    starty=y+save['scale']*1
    if debug:
        print(x,y,xstep,ystep,startx,starty)
        plt.figure(figsize=[5,5])
        plt.imshow(im)
        plt.title('start and end')
        plt.plot(startx,starty,'+w',markersize=20,mew=2)
        plt.plot(x,y,'+w',markersize=40,mew=2)
        plt.plot(startx+xstep*10,starty+ystep*10,'+w',markersize=20,mew=2)

    try:
        allvals = []
        readout = []
        x = startx
        y = starty
        
        #we step down the post (10 steps per 'tile')
        for step in range(11*10):
            readout.append(im[int(y),int(x)])
            x+=xstep*0.1
            y+=ystep*0.1

        readout = np.array(readout)
        
        if debug:
            plt.figure(figsize=[10,5])
            plt.plot(readout)
            print(readout)
        
        #find key points in the sequence (finding the search pattern 010110 = W | B | W | BB | W)
        #how much change equates to a change from 1->0 or 0->1
        stepreq = (np.max(readout)-np.min(readout))/5
        first_fall_in_search_pattern = np.where(((readout[3:]-readout[:-3])<-stepreq)[17:])[0][0]+17
        first_rise_in_search_pattern = np.where(((readout[3:]-readout[:-3])>stepreq)[first_fall_in_search_pattern:])[0][0]+first_fall_in_search_pattern
        second_fall_in_search_pattern = np.where(((readout[3:]-readout[:-3])<-stepreq)[first_rise_in_search_pattern:])[0][0]+first_rise_in_search_pattern
        second_rise_in_search_pattern = np.where(((readout[3:]-readout[:-3])>stepreq)[second_fall_in_search_pattern:])[0][0]+second_fall_in_search_pattern

        if debug:
            print(first_fall_in_search_pattern, first_rise_in_search_pattern, second_fall_in_search_pattern, second_rise_in_search_pattern)

        #now we know the two boundaries: 0|1011|0 separated by four tiles
        #we can compute the step_length.
        step_length = (second_rise_in_search_pattern - first_fall_in_search_pattern)/4
        
        if debug:
            print("Step length:")
            print(step_length)

        #find the start index in 'readout' of the top of the code
        #(the +3 is because we look 3 readout points apart)
        
        p = first_fall_in_search_pattern + 3 - (step_length * 2.5)
        vals = []

        #step through the code with the right step length, reading out values
        for pindx in np.arange(p,p+step_length*11,step_length):
            vals.append(readout[int(pindx)])

        vals = np.array(vals)
        
        #getting exact top/bottom coords...
        postcoords = [startx + p*xstep*0.1, starty + p*ystep*0.1,startx + (p+step_length*11)*xstep*0.1, starty + (p+step_length*11)*ystep*0.1]
         
    except IndexError:
        if debug:
            print("Index Error")
        return None, None
    
    #find middle value
    med = int((np.max(readout)+np.min(readout))/2)
    
    if debug:
        plt.figure(figsize=[10,5])
        plt.vlines(np.arange(p,p+step_length*11,step_length),0,255)
        plt.vlines(first_fall_in_search_pattern,0,255,'b')
        plt.vlines(second_fall_in_search_pattern,0,255,'b')
        plt.vlines(first_rise_in_search_pattern,0,255,'g')
        plt.vlines(second_rise_in_search_pattern,0,255,'g')
        plt.hlines(med*1.2,0,len(readout))
        plt.hlines(med*0.8,0,len(readout))
        plt.plot(readout)

    #convert to binary (with a NAN option if value lies too close to median)
    binaryvals = np.full_like(vals,np.NAN)
    binaryvals[vals>med*thresholds[0]] = 0
    binaryvals[vals<med*thresholds[1]] = 1
    
    #check 'black' is black enough
    if np.min(readout)>25:
        if debug:
            print("Minimum brightness too high (>25)")
        return None, None
    
    #check the signal is more than just noise (max>min+3 in original image)
    if np.max(readout)<np.min(readout)+30:
        if debug:
            print("insufficient signal - may be non-marker")
        return None, None
    
    if debug:
        print(binaryvals)
        
    #check all the bits were read properly
    if np.any(np.isnan(binaryvals)):
        if debug:
            print("NaN in array")
        return None, None
    
    #parity bit should mean an odd number of 1s.
    if np.sum(binaryvals)%2!=1: 
        if debug:
            print("Failed parity check")
        return None, None
    
    #confirm we read the search pattern correctly (010110)
    if (binaryvals[2]!=0) or (binaryvals[3]!=1) or (binaryvals[4]!=0) or (binaryvals[5]!=1) or (binaryvals[6]!=1) or (binaryvals[7]!=0):
        if debug:
            print("failed pattern check")
        return None, None
    
    #compute and return the value from the data stripes
    return ((binaryvals[0])*8 + (binaryvals[1]) * 4 + (binaryvals[8]) * 2 + (binaryvals[9]) * 1), postcoords


def gethash(obj):
    """
    Returns a 160 bit integer hash
    """
    return int(hashlib.sha1(obj).hexdigest(),16)

class Align():
    def build_observations(self):
        """
        Returns a numpy array of observations and their times.
        Parameters:
            - grouped_coords = the (time-indexed) dictionary of detected bees.
            - ignorepatches = a dictionary (indexed by camera) of lists,
               each element is the image coordinate of a place to ignore. Radius default = 100 pixels.
            - ignore_radius = 100 default
        Returns:
            observations an (Nx7) numpy array. Each row is one detected possible bee:
               - columns 0:3 = location of 'origin' i.e. camera
               - columns 3:6 = vector point along the line from the camera to the detected point
               - column 6 = the score given by the detection algorithm
            obstimes = (N) numpy array: the times (in seconds) of each detection
        """
        observations = []
        obstimes = []        
        
        for trigtime in sorted(self.grouped_coords.keys()):
            for obs in self.grouped_coords[trigtime]:
                if hasattr(self,'ignorepatches'):
                    if obs['cam'] in ignorepatches:
                        if len(self.ignorepatches[obs['cam']])>0:        
                            if np.min(np.linalg.norm(np.array(self.ignorepatches[obs['cam']])-np.array(obs['imgxy'])[None,:],axis=1))<100: continue
                #if obs['score']>2: continue
                obstimes.append(trigtime)
                origin = obs['origin']
                vect = obs['vect']
                vect = vect/np.linalg.norm(vect)
                observations.append(np.r_[origin,vect,obs['score']])
        observations = np.array(observations)
        obstimes = np.array(obstimes)
        return observations, obstimes

    def compute_retrodectect_coords(self,refresh_cache=False):
        """
        Runs the retrodetect tool on the images in the image_filenames list of filenames.
        Returns a list of potential bees (also stores in class object self.coords). Each item is a dictionary: 
          triggertime - the time of the photo, 
          xy - 2 item list of the image coordinates, 
          score - confidence in the dot
        More than one item can be returned for an image, conversely,
        when the algorithm fails to detect any, it won't add any items to
        the dictionary for the image.    
        """
        self.coords = {}
        for cam,image_fns in self.image_filenames.items():
            
            cachefn = "cache/coords_%d.pkl" % gethash((" ".join(image_fns)).encode("utf-8"))
            if not refresh_cache:
                try:
                    print("Trying to load from cache (%s) for camera %s" % (cachefn,cam))
                    self.coords[cam] = pickle.load(open(cachefn,'rb'))
                    continue
                except FileNotFoundError:
                    print("Cache not found (%s)" % cachefn)  
                    pass
            photo_list = []
            self.coords[cam] = []
            for fn in image_fns:
                try:
                    photoitem = np.load(fn,allow_pickle=True)
                except OSError:
                    continue #skip this one if we can't access it
                if photoitem is not None:
                    if photoitem['img'] is not None:
                        photoitem['img'] = photoitem['img'].astype(float)
                photo_list.append(photoitem)
                if len(photo_list)>20:
                    photo_list.pop(0)
                contact, found, _ = retrodetect.detectcontact(photo_list,len(photo_list)-1,Npatches=20)
                if contact is not None:
                    for con in contact:
                        if con['prediction']<8:
                            self.coords[cam].append({'triggertime':photo_list[-1]['record']['triggertime'], 'xy':[con['x'],con['y']], 'score':con['prediction']})
            print("Saving to cache (%s)" % cachefn)
            pickle.dump(self.coords[cam],open(cachefn,"wb"))
        return self.coords
    def build_ignore_patches(self, time_window=2.0,nest_item=None, nest_ignore_radius=200):
        """Other tagged bees, and other clutter can cause problems, here we are finding items
        that are probably such items. Ideally we should use some data when our bee-of-interest
        isn't in the photo. The process here is to find all candidate points for the first
        time_window seconds of data, and then ignore all detections within 100 pixels of these 
        points in future. However we don't ignore any point that is within 200px of the nest.

            grouped_coords = a time-indexed dictionary of detections
            time_window = how many seconds from the start of data collection to use
            nest_item = if not None, don't ignore an area nest_ignore_radius around the nest location.
                This should be a string identifying an item in the alignment.newitems dictionary.
        """
        if not hasattr(self,'ignorepatches'): self.ignorepatches = {}
        firsttrigtime = sorted(self.grouped_coords.keys())[0]
        for trigtime in sorted(self.grouped_coords.keys()):
            if trigtime>firsttrigtime+time_window: break #first second or two images...
            for gc in self.grouped_coords[trigtime]:
                if gc['cam'] not in self.ignorepatches:
                    self.ignorepatches[gc['cam']] = []
                if nest_item is not None:
                    if gc['cam'] in self.newitems[nest_item]['imgcoords']:
                        nestxy = self.newitems[nest_item]['imgcoords'][gc['cam']]
                        if (np.linalg.norm(np.array(gc['imgxy'])-np.array(nestxy)[None,:],axis=1))<nest_ignore_radius: continue
                self.ignorepatches[gc['cam']].append(gc['imgxy'])
        return self.ignorepatches
        
    def process_coords(self):
        """
        Combines the resutls from all the cameras into one dictionary, indexed by the time.
        Each element is a dictionary:
          - origin = the coordinates of the camera
          - vect = a vector pointing in the direction of the detected point
          - score = the score given to the point
          - imagxy = the position in the image
          - cam = which camera this is associated with.
        Returns them grouped (but also stores in class object as self.grouped_coords).    
        """
        self.grouped_coords = {}
        for cam in self.coords: 
            for j,c in enumerate(self.coords[cam]):
                if c['triggertime'] not in self.grouped_coords: self.grouped_coords[c['triggertime']] = []
                coord = c['xy']
                origin, vect = self.get_vector_pixel(self.newitems[cam],coord)  
                self.grouped_coords[c['triggertime']].append({'origin':origin,'vect':vect,'score':c['score'],'imgxy':c['xy'],'cam':cam})
        return self.grouped_coords
        
    def get_markers(self,image,threshold,debug=False):
        """
        Find the marker posts in the photo.
        Returns a dictionary of posts, e.g. one element might be:
        5.0: {'loc': (1804, 662),  --- mainly for internal use (ignore)
          'coord': array([1804, 1174]), --- coordinates of the post (top left corner)
          'scale': 7.142857142857144, --- scale (how many pixels wide roughly is the post)
          'angle': 5.555555555555557, --- the orientation of the post, i.e. it might not be vertical
          'score': 0.9434233, --- how confident it is in the detection
          'w': 21, --- mainly for internal use (ignore) [width and height of the search template]
          'h': 64},
        """
        
        #Remove top third of image as this usually just has clutter.
        shift = 0# int(image.shape[0]/3)
        im = image[shift:,:].copy()
        
        #scale 10x brighter
        im*=10
        im[im>255] = 255
        im[im<0]=0
        
        #the standard search code to find
        msg = [0,0,0,0,1,0,1,1,0]
        
        #build search template and mask
        inittempl = np.full([900,300],0).astype(np.float32)
        initmask = np.full([900,300],0).astype(np.float32)
        halfwidth = 60
        maskborderw = -10
        maskborderh = 0
        for step in range(3,9):
            inittempl[int(step*100):int(step*100+100),int(150-halfwidth):int(150+halfwidth)] = 255.0-(msg[step]*255.0) #int(step*slope+scale*0.5):int(scale*1.5+step*slope)] = 255.0-(msg[step]*255.0)
            initmask[int(step*100+maskborderh):int(step*100+100-maskborderh),int(150-halfwidth+maskborderw):int(150+halfwidth-maskborderw)] = 1
        inittempl[int(0*100+50):int(0*100+100),int(150-halfwidth):int(150+halfwidth)] = 220.0 #int(step*slope+scale*0.5):int(scale*1.5+step*slope)] = 255.0-(msg[step]*255.0)
        initmask[int(0*100+50):int(0*100+100),int(150-halfwidth):int(150+halfwidth)] = 1

        found = {}
        #run template matching at 15 different scales
        for n, scale in enumerate(np.linspace(0.05,0.2,15)): #(0.135,0.135,1):
            print(".", end="") #print("%d of %d" % (n,(15)))
            newsize = tuple((np.array([300,900]) * scale).astype(int))
            scaledtempl = cv2.resize(inittempl, newsize, interpolation = cv2.INTER_AREA)    
            scaledmask = cv2.resize(initmask, newsize, interpolation = cv2.INTER_AREA)    

            #run template matchinf for 10 different angles of pole 'tilt'
            for angle in np.linspace(-10,10,20):
                templ = rotate_image(scaledtempl,angle)
                mask = rotate_image(scaledmask,angle)
                
                #try match
                try:
                    res = cv2.matchTemplate(im,templ,cv2.TM_CCORR_NORMED,mask=np.trunc(mask+0.1))
                except Exception as e:

                    if debug: print(e,end="")
                    continue
                    
                w, h = templ.shape[::-1]
                
                #we look at those points in the result that have a high score
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    shifted_pt = np.array([pt[0],pt[1]+shift])
                    
                    #record location etc of the potential post
                    data = {'loc':pt, 'coord':shifted_pt, 'scale':scale*100, 'angle':angle,'score':res[pt[1],pt[0]],'w':w,'h':h}
                    data['im']=im #for debugging...
                    
                    #decode the post at that location
                    code, postcoords = getcode(data,im,debug=debug)
                    if postcoords is not None:
                        data['postcoords'] = postcoords
                        data['postcoords'][1] = data['postcoords'][1] + shift
                        data['postcoords'][3] = data['postcoords'][3] + shift
                    else:
                        data['postcoords'] = None
                    
                    #if decode was successful, we overwrite any other less well 'found' posts
                    if code is not None:
                        if code in found:
                            if found[code]['score']<res[pt[1],pt[0]]:
                                found[code]=data
                        else:
                            found[code]=data
                            
        #remove low quality guesses (that are nearby a good one)               
        rems = []
        for f in found:
            for g in found:
                if f==g: continue
                if np.sum((np.array(found[f]['loc'])-np.array(found[g]['loc']))**2)<100:
                    if found[f]['score']>found[g]['score']:
                        rems.append(g)
                    else:
                        rems.append(f)
        for r in set(rems):
            found.pop(r)
        
        print("Found %d." % len(found))
        return found

    def __init__(self,config_filename,image_filenames):
        """
        config_filename = a json file with a dictionary of:
            - dists: a list of dictionaries, containing: 'from', 'to', and 'dist'.
            - items: a dictionary of items (same as in 'from' and 'to', each with a 'coords' element).
            e.g.:
            {
                "dists": [ <--- rough distance between items
                    {
                        "from": "cam4",
                        "to": "cam1",
                        "dist": 5.8
                    },....
                ]
                "items": { <--- list of items
                        "cam4": {
                            "height": 1.87, <--- height above ground
                            "angle": 6.31, <--- optional, for cameras, rough guess about which way they're facing
                            "coords": [ <---can be a 3-item list of a coord, or a string pointing to another item's location.
                                0,
                                12,
                                0
                            ]
                         "imgcoords": {"cam1":[1768.0,1292.0],"cam2":[1242.0, 947.0],... } <--- location of the item in images taken by the cameras.
                        },...
                }
            }

        image_filenames = a dictionary of lists of image filenames.
            - each element of the dictionary is a camera (e.g. cam1, cam2, cam3, cam4).
            - each one has an (ordered) list of filenames for that camera.
        """
        config = json.load(open(config_filename,'r'))
        self.config_filename = config_filename
        self.dists = config['dists']
        self.items = config['items']
        if 'ignorepatches' in config:
            self.ignorepatches = config['ignorepatches']
        self.image_filenames = image_filenames

    def optimise_positions(self,check_result_threshold=1,allow_random_initialisation=False):
        """
        Optimises the locations
        """
        print("Optimising Locations")
        for it in self.items:
            if 'coords' not in self.items[it]:
                if not allow_random_initialisation: assert False, "Missing initial coordinate in %s" % it
                self.items[it]['coords'] = np.random.randn(3)*10
        for i in range(1000):
            for d in self.dists:
                #how much the distance needs to increase to reach target
                if (d['from'] not in self.items) or (d['to'] not in self.items):
                    #print("Warning: %s or %s not found in items. This distance wasn't used.")
                    continue
                from_minus_to = np.array(self.items[d['from']]['coords'])-np.array(self.items[d['to']]['coords'])
                lenfromto = np.sqrt(np.sum(from_minus_to**2))
                delta = d['dist']-lenfromto
                from_minus_to=from_minus_to/lenfromto
                self.items[d['from']]['coords']+=from_minus_to*delta*0.1
                self.items[d['to']]['coords']-=from_minus_to*delta*0.1
                
                #check that the distances are all ok...
                if i==999:
                    assert np.abs(d['dist']-lenfromto)<check_result_threshold, "Unable to find consistency in distance data. In particular %s failed (found %0.1f distance)" % (str(d), lenfromto)
        for it in self.items:
            if type(self.items[it]['coords'])==str:
                #TODO Should probably check if self.items[it]['coords'] is in self.items, and that it has a 'coords' item.
                self.items[it]['coords'] = deepcopy(self.items[self.items[it]['coords']]['coords'])
        
    def load_photos_for_alignment(self,flash=False):
        """Loads a photo for each camera (finds ones that either are or aren't flash photos)"""
        
        self.cam_photos = {}
        for photo_fn_idx in self.image_filenames:
            temp = []
            m = 0
            for fn in self.image_filenames[photo_fn_idx][:10]:
                a = np.load(fn,allow_pickle=True)
                #m = np.mean(a['img'].astype(np.float32)))
                if a['img'] is not None:
                    temp.append(a['img'].astype(np.float32))
            mean = np.mean([np.mean(img) for img in temp])
            for img in temp:
                if not flash:
                    if np.mean(img)<mean-0.1:
                        self.cam_photos[photo_fn_idx] = img
                        break
                else:
                    if np.mean(img)>mean+0.1:
                        self.cam_photos[photo_fn_idx] = img
                        break

        
    #def gethash(self,v):
    #    """Pass image 'v', returns integer hash"""
    #    return int(np.sum((v.astype(float)*(np.arange(0,len(v))[:,None]))+(v.astype(float)*(np.arange(0,v.shape[1])[None,:]))))
    

    def find_markers(self,refresh_cache=False,debug=False,threshold=0.91,photo_indicies=None):
        """
        Find markers in self.cam_photos
        - refresh_cache - set to True to recompute
        - photo_indicies - set to a list of indicies
        """
        if not hasattr(self,'found'):
            self.found = {}
        fnd = {}
        self.shift = int(next(iter(self.cam_photos.values())).shape[0]/3) #this is for debugging, might change value - not cached...
        
        if photo_indicies is None:
            photo_indicies = self.cam_photos.keys()
        for photo_idx in photo_indicies:
            cache_fn = "cache/found_cache_%d.pkl" % gethash(self.cam_photos[photo_idx])
            try:
                print("Trying to load from cache (%s)" % cache_fn)
                if refresh_cache:
                    raise FileNotFoundError #hack to trigger a recalc.
                f = pickle.load(open(cache_fn,'rb'))
                
            except FileNotFoundError:
                print("Cache not found (%s)" % cache_fn)                
                f = self.get_markers(self.cam_photos[photo_idx],threshold,debug=debug)
                print("Saving to cache (%s)" % cache_fn)
                pickle.dump(f,open(cache_fn,'wb'))
            fnd[photo_idx] = f
            
            for f in fnd[photo_idx]:
                if fnd[photo_idx][f]['score']<0.91: continue #temp: removes low quality guesses
                
                if photo_idx not in self.found:
                    self.found[photo_idx] = {}
                    
                if f in self.found[photo_idx]:
                    if fnd[photo_idx][f]['score']>self.found[photo_idx][f]['score']:
                        self.found[photo_idx][f] = fnd[photo_idx][f]
                else:
                    self.found[photo_idx][f] = fnd[photo_idx][f]

        
    
    def draw_found(self,cam,imthresh=30,draw_3d_location=False):
        im = self.cam_photos[cam]
        found_list = self.found[cam]
            
        plt.imshow(im,cmap='gray')
        #imthresh = 30 #np.quantile(im[600:,:],0.95)
        for f in found_list:
            fnd = found_list[f]
            ax = plt.gca()
            col = 'y'
            plt.text(fnd['coord'][0],fnd['coord'][1],"%d %d" % (f,int(fnd['score']*100)),color=col,fontsize=20)
            #plt.plot([fnd['postcoords'][0],fnd['postcoords'][2]],np.array([fnd['postcoords'][1],fnd['postcoords'][3]]),'y-')
            plt.plot([fnd['postcoords'][0],fnd['postcoords'][2]],np.array([fnd['postcoords'][1],fnd['postcoords'][3]]),'y-')
            plt.plot([fnd['postcoords'][0],fnd['postcoords'][2]],np.array([fnd['postcoords'][1],fnd['postcoords'][3]]),'y+',markersize=10,mew=1)
            #rect = patches.Rectangle(fnd['coord'], fnd['w'], fnd['h'], linewidth=1, edgecolor=col, facecolor='none')
            #ax.add_patch(rect)
        plt.clim([0,imthresh])
        for it, v in self.newitems.items():
            if 'imgcoords' in v:
                if cam in v['imgcoords']:
                    plt.plot(v['imgcoords'][cam][0],v['imgcoords'][cam][1],'xy',markersize=20)
                    plt.text(v['imgcoords'][cam][0],v['imgcoords'][cam][1],it,color='yellow')        
        if draw_3d_location:
            for it in self.newitems:
                imgxy = self.get_pixel_loc(self.newitems[cam],self.newitems[it]['coords'])
                if (imgxy[0,0]>0) and (imgxy[0,0]<2048) and (imgxy[0,1]>0) and (imgxy[0,1]<2048*0.75):
                    plt.plot(imgxy[0,0],imgxy[0,1],'b+',markersize=30,mew=3)
                    plt.text(imgxy[0,0],imgxy[0,1],it,color='b')
                    if 'imgcoords' in self.newitems[it]:
                        if (cam in self.newitems[it]['imgcoords']):
                            knownxy = self.newitems[it]['imgcoords'][cam]
                            plt.plot(knownxy[0],knownxy[1],'yx')
    
    def tryangle(self,params,cam):
        a = params[0]
        fov = params[1]
        toterr = 0
        for f in self.found[cam]:
            fnd = self.found[cam][f]
            p = (fov/2) * (fnd['coord'][0]-2048/2)/(2048/2)
            d = np.array(self.items["m%02d" % f]['coords'])-np.array(self.items[cam]['coords'])
            d = d/np.sqrt(np.sum(d**2))
            toterr+=((np.cos(-p+a)-d[0])**2 + (np.sin(-p+a)-d[1])**2)
        return toterr    
    
    def estimate_fov(self):
        """
        Estimates the FOV of the optics, from the locations of the markers/camera and their angles in the image
        Returns the median from all the cameras and a dictionary of optimiser objects
        
        Also populates items with angles.
        
        """
        fovs = []
        angs = []
        ret = {}
        for cam in self.found:
            fov = np.pi/4
            a = 0
            opt = optimize.minimize(lambda x: self.tryangle(x,cam),[a,fov])
            #print(opt['x'],opt['hess_inv'])
            ret[cam] = opt
            fovs.append(opt['x'][1])
            angs.append(opt['x'][0])
            self.items[cam]['angle'] = opt['x'][0]
        self.fov = np.median(fovs)
        return self.fov, ret

    def get_pixel_loc(self, cam, markercoords):
        p = np.array(markercoords - cam['coords'])
        r1 = R.from_euler('z', -cam['angle'], degrees=False) #yaw
        r2 = R.from_euler('Y', -cam['pitch'], degrees=False) #pitch (intrinsic rotation around y axis)    
        r3 = R.from_euler('X', -cam['roll'], degrees=False) #roll (intrinsic rotation around x axis)    

        pvec = r3.apply(r2.apply(r1.apply(p)))
        if len(pvec.shape)==1:
            pvec = pvec[None,:]
        res = np.array([1024+1024*(-pvec[:,1]/pvec[:,0])/self.hfovw,(1024+1024*(pvec[:,2]/pvec[:,0]/self.vfovw))*0.75]).T
        #assert np.all(np.array(self.old_get_pixel_loc(cam,markercoords))==res)
        return res
        
    def old_get_pixel_loc(self, cam, markercoords):
        p = np.array(markercoords - cam['coords'])
        r1 = R.from_euler('z', -cam['angle'], degrees=False) #yaw
        r2 = R.from_euler('Y', -cam['pitch'], degrees=False) #pitch (intrinsic rotation around y axis)    
        r3 = R.from_euler('X', -cam['roll'], degrees=False) #roll (intrinsic rotation around x axis)    
        
        pvec = r3.apply(r2.apply(r1.apply(p)))
        #print(cam['angle'],p,pvec)
        if np.abs(pvec[0])<0.01:
            #print(pvec[0])
            pvec[0] = 0.01
        return 1024+1024*(-pvec[1]/pvec[0])/self.hfovw,(1024+1024*(pvec[2]/pvec[0]/self.vfovw))*0.75       
    
    def get_vector_pixel(self,cam,pixel):
        p = np.zeros(3)
        p[0] = 1
        p[1] = -self.hfovw*(pixel[0]-1024)/1024
        p[2] = self.vfovw*((pixel[1]/0.75)-1024)/1024
        r1 = R.from_euler('z', cam['angle'], degrees=False) #yaw
        r2 = R.from_euler('Y', cam['pitch'], degrees=False) #pitch (intrinsic rotation around y axis)
        r3 = R.from_euler('X', cam['roll'], degrees=False) #roll (intrinsic rotation around x axis)        
        pvec = r1.apply(r2.apply(r3.apply(p)))
        return cam['coords'],pvec
    
    

    def compute_error(self,params,findworst=False):
        """
        Using the values in params, assign new coords etc to the newitems,
        then compute the sum squared error of their locations when 'rendered' back to the camera images.
        
        If findworst is set to true, it returns the distance of the least accurate fit.
        """
        err = 0
        i=0
        self.fov = params[i]
        i+=1
        self.hfovw = np.tan(self.fov/2)
        self.vfovw = np.tan(0.75*self.fov/2)

        for n in self.newitems:
            adjustpos = False
            if 'imgcoords' in self.newitems[n]:
                if len(self.newitems[n]['imgcoords'])>0:
                    adjustpos = True
            if n[:3]=='cam': adjustpos = True
            if adjustpos:
                
                if n[-6:]=='bottom':
                    self.newitems[n]['coords'][2] = self.newitems[n[:4]+'top']['coords'][2]+0.516
                    self.newitems[n]['coords'][0] = self.newitems[n[:4]+'top']['coords'][0]+params[i]
                    i+=1
                    self.newitems[n]['coords'][1] = self.newitems[n[:4]+'top']['coords'][1]+params[i]
                    i+=1
                else:
                    self.newitems[n]['coords'][0] = params[i]
                    i+=1
                    self.newitems[n]['coords'][1] = params[i]
                    i+=1
                    self.newitems[n]['coords'][2] = params[i]
                    i+=1


            if n[:3]=='cam':
                self.newitems[n]['roll'] = params[i]
                i+=1
                self.newitems[n]['pitch'] = params[i]
                i+=1
                self.newitems[n]['angle'] = params[i]
                i+=1

        for n in self.newitems:
            if 'imgcoords' not in self.newitems[n]: continue
            for cam in self.newitems[n]['imgcoords']:
                pred_pos = self.get_pixel_loc(self.newitems[cam],self.newitems[n]['coords'])
                act_pos = self.newitems[n]['imgcoords'][cam]
                if 'weight' in self.newitems[n]: 
                    w = self.newitems[n]['weight']
                else:
                    w = 1.0
                if findworst:
                    err = max(err,np.sqrt(np.sum((np.array(pred_pos)-np.array(act_pos))**2)))
                else:
                    err += w * np.sum((np.array(pred_pos)-np.array(act_pos))**2)
        return err


    def build_newitems(self):
        """
        Call this to build 'newitems' from 'items', then call generate_alignment.
        """
        self.newitems = {}
        for it in self.items:
            if (it[0]=='m'):
                if (len(it)==3): #marker (not _top or _bottom, we deal with these next).
                    self.newitems[it+'_top'] = deepcopy(self.items[it])
                    self.newitems[it+'_bottom'] = deepcopy(self.items[it])
                    self.newitems[it+'_top']['imgcoords'] = {}
                    self.newitems[it+'_bottom']['imgcoords'] = {}
                    for cam in self.found:
                        for f in self.found[cam]:
                            markerid = "m%02d" % int(f)
                            if it == markerid:
                                self.newitems[it+'_top']['imgcoords'][cam] = deepcopy(self.found[cam][f]['postcoords'][:2])
                                self.newitems[it+'_bottom']['imgcoords'][cam] = deepcopy(self.found[cam][f]['postcoords'][2:])
            else:
                self.newitems[it] = deepcopy(self.items[it])

        #Previously I assumed that the markers would need to be split into top and bottom
        #by this code, but we might also want to include img coords of those that it can't find.
        for it in self.items:
            if len(re.findall('m[0-9]{2}_',it))>0: #is this landmark m??_something?
                if it not in self.newitems:
                    self.newitems[it] = {'imgcoords':{}}
                if 'imgcoords' in self.items[it]: self.newitems[it]['imgcoords'] = {**self.items[it]['imgcoords'],**self.newitems[it]['imgcoords']}
                
        for it in self.newitems:
            if it[:3]=='cam':
                self.newitems[it]['pitch'] = 0
                      
    def generate_alignment(self,maxiters = 1000,refresh_cache=False,printlabels=False):
        """
        Try to find best positions and orientations of markers and cameras
        """
        #these are the horizontal and vertial 'fov' points on the projection plane (z=1)
        self.hfovw = np.tan(self.fov/2)
        self.vfovw = np.tan(0.75*self.fov/2)

        #combines both the config file and the fns of the images
        hashnumber = gethash((self.config_filename+" "+" ".join([fns[0] for cam, fns in self.image_filenames.items()])).encode("utf-8"))
        cachefn = "cache/camera_alignment_cache_%d.pkl" % hashnumber
        #cachefn = "cache/camera_alignment_cache_%d.pkl" % gethash(self.config_filename.encode("utf-8"))
        if not refresh_cache:
            try:
                print("Trying to load from cache (%s)" % cachefn)
                optx = pickle.load(open(cachefn,'rb'))
                worst = self.compute_error(optx,findworst=True) #updates parameters to this optimum result
                print("Worst position %d pixels away." % int(worst))
                return optx
            except FileNotFoundError:            
                print("Cache not found (%s)" % cachefn)                
                pass

        bounds = []
        
        bounds.append([self.fov-0.4,self.fov+0.4])
        boundstrings = []
        boundstrings.append("fov")
        movementallowed = 1.6
        
        for n in self.newitems:
            adjustpos = False
            if 'imgcoords' in self.newitems[n]:
                if len(self.newitems[n]['imgcoords'])>0:
                    adjustpos = True
            if n[:3]=='cam': adjustpos = True
            if adjustpos:
                
                if n[-6:]=='bottom':
                    bounds.append([-0.1,0.1])
                    boundstrings.append(n+"x")
                    bounds.append([-0.1,0.1])
                    boundstrings.append(n+"y")
                else:
                    if 'bounds' in self.newitems[n]:
                        bounds.append([self.newitems[n]['coords'][0]-self.newitems[n]['bounds'],self.newitems[n]['coords'][0]+self.newitems[n]['bounds']])
                        boundstrings.append(n+"x")
                        bounds.append([self.newitems[n]['coords'][1]-self.newitems[n]['bounds'],self.newitems[n]['coords'][1]+self.newitems[n]['bounds']])
                        boundstrings.append(n+"y")
                        bounds.append([-2,2]) #move up or down by 1.5m
                        boundstrings.append(n+"z")
                    else:                                                
                        bounds.append([self.newitems[n]['coords'][0]-movementallowed,self.newitems[n]['coords'][0]+movementallowed])
                        boundstrings.append(n+"x")
                        bounds.append([self.newitems[n]['coords'][1]-movementallowed,self.newitems[n]['coords'][1]+movementallowed])
                        boundstrings.append(n+"y")
                        bounds.append([-2,2]) #move up or down by 1.5m
                        boundstrings.append(n+"z")




            if n[:3]=='cam':
                bounds.append([-0.1,0.1]) #roll (limited)
                boundstrings.append("roll")
                bounds.append([-0.5,0.2]) #pitch (mostly pitch down I think?)
                boundstrings.append("pitch")
                bounds.append([self.items[n]['angle']-1.5,self.items[n]['angle']+1.5])
                #bounds.append([-np.pi*10,np.pi*10])
                boundstrings.append("yaw")
        len(bounds)

#        opt = optimize.minimize(self.compute_error,np.zeros(len(bounds)),bounds=bounds,options={'maxiter':maxiters},callback=lambda x: print("%d" % int(self.compute_error(x)),end=", "))
        
        global count
        count = 0
        def showprog(x):
            global count
            count = count + 1 
            if (count%10==0): 
                print("%0.0f" % self.compute_error(x,findworst=True),end=" ")
        opt = optimize.minimize(self.compute_error,np.mean(np.array(bounds),1),bounds=bounds,options={'maxiter':maxiters},callback=showprog) 
        
        
        #,callback=lambda x: print(x)) #print("%d" % int(self.compute_error(x)),end=", "))
        print(" ")
        print("Saving to cache (%s)" % cachefn)
        worst = self.compute_error(opt['x'],findworst=True) #updates parameters to this optimum result
        print("Worst position %d pixels away." % int(worst))
        pickle.dump(opt['x'],open(cachefn,'wb'))
        self.bounds = bounds
        self.boundstrings = boundstrings
        
        print("Optimiser complete")
        
        #provides a warning to the user if there's a potential problem
        msg = ""
        for i,(bound,x,st) in enumerate(zip(bounds,opt['x'],boundstrings)):
            ratio = (x-bound[0])/(bound[1]-bound[0])
            if (ratio<0.03) or (ratio>0.97):
                msg += "%20s: %6.2f to %6.2f  %6.2f\n" % (st,bound[0],bound[1],x)
        if len(msg)>0:
            print("Some parameters we close to their bound limits, this may indicate")
            print("initial values (that are used to select the bounds) were incorrect.\n")
            print("           parameter   ------bound----   value")
        print(msg)
                
        return opt['x']
        
    def draw_path_on_photos(self,obstimes,strajs,camid):
        """
        Draw the path on the photos from our alignment.
        Mean path = blue line
        Standard deviation = yellow circles.
        Time = yellow numbers (seconds)
        """
        photo = self.cam_photos[camid]
        meanpath = np.mean(strajs,1)
        plt.imshow(photo,cmap='gray')
        plt.clim([0,15])

        mp_pixels =self.get_pixel_loc(self.newitems[camid],meanpath)

        keep = []
        for i,mp in enumerate(mp_pixels[:-1]):

            particle_pixels = self.get_pixel_loc(self.newitems[camid],strajs[i+1,:,:])
            partstd = np.mean(np.std(particle_pixels,0)) 
            if partstd>100: continue
            keep.append(mp_pixels[i,:])
            plt.plot(mp_pixels[i:(i+2),0],mp_pixels[i:(i+2),1],'b-x',lw=(200/partstd)-1)
            if i%5 ==0:
                plt.text(mp[0],mp[1],"%0.1f" % (obstimes[i]-obstimes[0]),color='yellow',fontsize=20)
                plt.gca().add_artist(plt.Circle(mp, partstd,fill=False,color='y'))
     
        nestxy = self.newitems['nestfrontleft']['imgcoords'][camid]    
        keep = np.array(keep)
        margin = 250
        plt.xlim([np.min(keep,0)[0]-margin,np.max(keep,0)[0]+margin])
        plt.ylim([np.max(keep,0)[1]+margin,np.min(keep,0)[1]-margin])        
