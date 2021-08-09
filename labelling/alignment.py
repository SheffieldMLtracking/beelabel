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

def getcode(save,im,debug=False,shift=None,thresholds = [1.25,0.75]):
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




class Align():
    def get_markers(self,image):
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
        shift = int(image.shape[0]/3)
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
                except:
                    continue
                    
                w, h = templ.shape[::-1]
                
                #we look at those points in the result that have a high score
                threshold = 0.91 #0.88 #0.91
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    shifted_pt = np.array([pt[0],pt[1]+shift])
                    
                    #record location etc of the potential post
                    data = {'loc':pt, 'coord':shifted_pt, 'scale':scale*100, 'angle':angle,'score':res[pt[1],pt[0]],'w':w,'h':h}
                    data['im']=im #for debugging...
                    
                    #decode the post at that location
                    code, postcoords = getcode(data,im)
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
                "dists": [
                    {
                        "from": "cam4",
                        "to": "cam1",
                        "dist": 5.8
                    },....
                ]
                "items": {
                        "cam4": {
                            "height": 1.87,
                            "angle": 6.31,
                            "coords": [
                                0,
                                12,
                                0
                            ]
                        },...
                }
            }

        image_filenames = a dictionary of lists of image filenames.
            - each element of the dictionary is a camera (e.g. cam1, cam2, cam3, cam4).
            - each one has an (ordered) list of filenames for that camera.
        """
        config = json.load(open(config_filename,'r'))
        self.dists = config['dists']
        self.items = config['items']
        self.image_filenames = image_filenames

    def optimise_positions(self,check_result_threshold=1,allow_random_initialisation=False):
        """
        Optimises the locations
        """
        
        for it in self.items:
            if 'coords' not in self.items[it]:
                if not allow_random_initialisation: assert False, "Missing initial coordinate in %s" % it
                self.items[it]['coords'] = np.random.randn(3)*10

        for i in range(1000):
            for d in self.dists:
                #how much the distance needs to increase to reach target
                from_minus_to = np.array(self.items[d['from']]['coords'])-np.array(self.items[d['to']]['coords'])
                lenfromto = np.sqrt(np.sum(from_minus_to**2))
                delta = d['dist']-lenfromto
                from_minus_to=from_minus_to/lenfromto
                self.items[d['from']]['coords']+=from_minus_to*delta*0.1
                self.items[d['to']]['coords']-=from_minus_to*delta*0.1
                
                #check that the distances are all ok...
                if i==999:
                    assert np.abs(d['dist']-lenfromto)<check_result_threshold, "Unable to find consistency in distance data. In particular %s failed (found %0.1f distance)" % (str(d), lenfromto)
                    
    def load_photos_for_alignment(self,flash=False):
        """Loads a photo for each camera (finds ones that either are or aren't flash photos)"""
        
        self.cam_photos = {}
        for photo_fn_idx in self.image_filenames:
            temp = []
            m = 0
            for fn in self.image_filenames[photo_fn_idx][:10]:
                a = np.load(fn,allow_pickle=True)
                #m = np.mean(a['img'].astype(np.float32)))
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
    def gethash(self,v):
        """Pass image 'v', returns integer hash"""
        return int(np.sum((v.astype(float)*(np.arange(0,len(v))[:,None]))+(v.astype(float)*(np.arange(0,v.shape[1])[None,:]))))

    def find_markers(self,refresh_cache=False,debug=False,photo_indicies=None):
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
            cache_fn = "found_cache_%d.pkl" % self.gethash(self.cam_photos[photo_idx])
            try:
                print("Trying to load from cache (%s)" % cache_fn)
                if refresh_cache:
                    raise FileNotFoundError #hack to trigger a recalc.
                f = pickle.load(open(cache_fn,'rb'))
                
            except FileNotFoundError:
                print("Cache not found (%s)" % cache_fn)                
                f = self.get_markers(self.cam_photos[photo_idx])
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

        
    
    def draw_found(self,im,found_list,imthresh=30):
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
        #plt.colorbar()
    
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

    
    def get_pixel_loc(self, cam, marker):
        p = np.array(marker['coords'] - cam['coords'])
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
    
    

    def compute_error(self,params):
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
                pred_pos = self.get_pixel_loc(self.newitems[cam],self.newitems[n])
                act_pos = self.newitems[n]['imgcoords'][cam]
                if 'weight' in self.newitems[n]: 
                    w = self.newitems[n]['weight']
                else:
                    w = 1.0
                err += w * np.sum((np.array(pred_pos)-np.array(act_pos))**2)
        return err


    def build_newitems(self):
        """
        Call this to build 'newitems' from 'items', then call generate_alignment.
        """
        self.newitems = {}
        for it in self.items:
            if it[0]=='m': #marker
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

        for it in self.newitems:
            if it[:3]=='cam':
                self.newitems[it]['pitch'] = 0  
                      
    def generate_alignment(self,maxiters = 1000):
        """
        Try to find best positions and orientations of markers and cameras
        """
        #these are the horizontal and vertial 'fov' points on the projection plane (z=1)
        self.hfovw = np.tan(self.fov/2)
        self.vfovw = np.tan(0.75*self.fov/2)

        bounds = []
        
        bounds.append([self.fov-0.2,self.fov+0.2])
        
        for n in self.newitems:
            adjustpos = False
            if 'imgcoords' in self.newitems[n]:
                if len(self.newitems[n]['imgcoords'])>0:
                    adjustpos = True
            if n[:3]=='cam': adjustpos = True
            if adjustpos:
                
                if n[-6:]=='bottom':
                    bounds.append([-0.1,0.1])
                    bounds.append([-0.1,0.1])
                else:
                    if 'bounds' in self.newitems[n]:
                        bounds.append([self.newitems[n]['coords'][0]-self.newitems[n]['bounds'],self.newitems[n]['coords'][0]+self.newitems[n]['bounds']])
                        bounds.append([self.newitems[n]['coords'][1]-self.newitems[n]['bounds'],self.newitems[n]['coords'][1]+self.newitems[n]['bounds']])
                        bounds.append([-2,2]) #move up or down by 1.5m
                    else:                                                
                        bounds.append([self.newitems[n]['coords'][0]-1,self.newitems[n]['coords'][0]+1])
                        bounds.append([self.newitems[n]['coords'][1]-1,self.newitems[n]['coords'][1]+1])
                        bounds.append([-2,2]) #move up or down by 1.5m




            if n[:3]=='cam':
                bounds.append([-0.1,0.1]) #roll (limited)
                bounds.append([-0.5,0.2]) #pitch (mostly pitch down I think?)
                bounds.append([self.items[n]['angle']-1.5,self.items[n]['angle']+1.5])
        len(bounds)

        opt = optimize.minimize(self.compute_error,np.zeros(len(bounds)),bounds=bounds,options={'maxiter':maxiters},callback=lambda x: print(self.compute_error(x)))
        return opt
