import numpy
import scipy.linalg

import param

from colorhacks.colorfns import threeDdot, \
     hsv_to_rgb, rgb_to_hsv, \
     xyz_to_lab, lab_to_xyz, lab_to_lch, lch_to_lab

# started from 
# http://projects.scipy.org/scipy/browser/trunk/Lib/sandbox/image/color.py?rev=1698

whitepoints = {'CIE A': ['Normal incandescent', 0.4476, 0.4074],
               'CIE B': ['Direct sunlight', 0.3457, 0.3585],
               'CIE C': ['Average sunlight', 0.3101, 0.3162],
               'CIE E': ['Normalized reference', 1.0/3, 1.0/3],
               'D50' : ['Bright tungsten', 0.3457, 0.3585],
               'D55' : ['Cloudy daylight', 0.3324, 0.3474],
               'D65' : ['Daylight', 0.312713, 0.329016],
               'D75' : ['?', 0.299, 0.3149],
               'D93' : ['low-quality old CRT', 0.2848, 0.2932]
               }

def triwhite(chrwhite):
    x,y = chrwhite
    X = float(x) / y
    Y = 1.0
    Z = (1-x-y)/y
    return X,Y,Z

for key in whitepoints.keys():
    whitepoints[key].append(triwhite(whitepoints[key][1:]))


transforms = {}


################################################################
# sRGB
# CEBALERT: add reference
transforms['srgb'] = {}
transforms['srgb']['D65'] = srgbD65 = {}

srgbD65['rgb_from_xyz'] = numpy.array([[3.2410,-1.5374,-0.4986],
                                       [-0.9692,1.8760,0.0416],
                                       [0.0556,-0.204,1.0570]])
srgbD65['xyz_from_rgb'] = scipy.linalg.inv(srgbD65['rgb_from_xyz'])
################################################################


################################################################
# spLMS
transforms['splms'] = {}
transforms['splms']['D65'] = splmsD65 = {}

# CEBALERT: reckon this matrix is wrong - just pasted as placeholder
# (need to get from my code)

# Guth (1980) - SP; L, M, and S normalized to one)
splmsD65['lms_from_xyz'] = numpy.array([[0.2435, 0.8524, -0.0516],
                                        [-0.3954, 1.1642, 0.0837],
                                        [0, 0, 0.6225]])

splmsD65['xyz_from_lms'] = scipy.linalg.inv(splmsD65['lms_from_xyz'])
################################################################


# This started off general but ended up being useful only
# for the specific transforms I wanted to do.
class ColorSpace(param.Parameterized):
    whitepoint = param.String(default='D65')

    transforms = param.Dict()

    input_limits = param.NumericTuple((0.0,1.0))

    output_limits = param.NumericTuple((0.0,1.0))

    output_clip = param.ObjectSelector(default='silent',
                                       objects=['silent','warn','error'])

    dtype = param.Parameter(default=numpy.float32)

    def _get_shape(self,a):        
        if hasattr(a,'shape') and a.ndim>0: # i.e. really an array, I hope
            return a.shape
        else:
            # also support e.g. tuples
            try:
                length = len(a)
                return (length,)
            except TypeError:            
                return None

    def _put_shape(self,a,shape):
        if shape is None:
            return self.dtype(a)
        else:
            a.shape = shape
            return a
    
    def _prepare_input(self,a,min_,max_):
        in_shape = self._get_shape(a)
        a = numpy.array(a,copy=False,ndmin=3,dtype=self.dtype)
        if a.min()<min_ or a.max()>max_:
            raise ValueError('out of limits')
        return a, in_shape
        
    def _clip(self,a,min_limit,max_limit,action='silent'):
        if action=='error':
            if a.min()<min_limit or a.max()>max_limit:
                raise ValueError
        elif action=='warn':
            if a.min()<min_limit or a.max()>max_limit:
                self.warning('outside limits')

        a.clip(min_limit,max_limit,out=a)
                    
    def _threeDdot(self,M,a):
        # b = Ma        
        a, in_shape = self._prepare_input(a,*self.input_limits)
        b = threeDdot(M,a)
        self._clip(b,*self.output_limits,action=self.output_clip)
        self._put_shape(b,in_shape)
        return b

    def _ABC_to_DEF_by_fn(self,ABC,fn):
        ABC, in_shape = self._prepare_input(ABC,*self.input_limits)
        DEF = fn(ABC)
        self._clip(DEF,*self.output_limits,action=self.output_clip)
        self._put_shape(DEF, in_shape)
        return DEF

    # CEBALERT: I meant to wrap these to use paramoverrides (e.g. to
    # allow rgb_to_hsv(RGB,output_clip='error') ) but never got round
    # to it.
    # CEBALERT: could cut down boilerplate by generating.

    def rgb_to_hsv(self,RGB):
        return self._ABC_to_DEF_by_fn(RGB,rgb_to_hsv)

    def hsv_to_rgb(self,HSV):
        return self._ABC_to_DEF_by_fn(HSV,hsv_to_rgb)
        
    def xyz_to_rgb(self,XYZ):
        return self._threeDdot(
            self.transforms[self.whitepoint]['rgb_from_xyz'], XYZ)

    def rgb_to_xyz(self,RGB):
        return self._threeDdot(
            self.transforms[self.whitepoint]['xyz_from_rgb'], RGB)

    def xyz_to_lch(self, XYZ):
        return xyz_to_lch(XYZ,self.whitepoint)

    def lch_to_xyz(self,LCH):
        return lch_to_xyz(LCH,self.whitepoint)

    def lch_to_rgb(self,LCH):
        return self.xyz_to_rgb(self.lch_to_xyz(LCH))

    def xyz_to_lms(self,XYZ):
        return self._threeDdot(
            self.transforms[self.whitepoint]['lms_from_xyz'], XYZ)

    def lms_to_xyz(self,LMS):
        return self._threeDdot(
            self.transforms[self.whitepoint]['xyz_from_lms'], LMS)

    def lch_to_lms(self,LCH):
        return self.xyz_to_lms(self.lch_to_xyz(LCH))


class sRGB(ColorSpace):

    transforms = param.Dict(default=transforms['srgb'])

    @staticmethod
    def _gamma(RGB):
        return 12.92*RGB*(RGB<=0.0031308) + ((1+0.055)*RGB**(1/2.4) - 0.055) * (RGB>0.0031308)

    @staticmethod
    def _ungamma(RGB):
        return RGB/12.92*(RGB<=0.04045) + (((RGB+0.055)/1.055)**2.4) * (RGB>0.04045)

    def xyz_to_gammargb(self,XYZ):
        return self._gamma(self.xyz_to_rgb(XYZ))
    
    def gammargb_to_xyz(self,RGB):
        return self.rgb_to_xyz(self._ungamma(RGB))

    def lch_to_gammargb(self,LCH):
        return self._gamma(self.lch_to_rrgb(LCH))

    # For HSV, whether you transform from gamma corrected RGB or
    # linear RGB is not defined as far as I know.    
    def gammargb_to_hsv(self,RGB):
        return self.rgb_to_hsv(RGB)

    def hsv_to_gammargb(self,HSV):
        return self.hsv_to_rgb(HSV)



class spLMS(ColorSpace):

    transforms = param.Dict(default=transforms['splms'])


class TopoColorConverter(param.Parameterized):

    # CEBALERT: should be class selector
    colorspace = param.Parameter(default=ColorSpace())

    analysis_space = param.ObjectSelector(
        default='HSV',
        objects=['HSV','LCH'])
    
    receptor_space = param.ObjectSelector(
        default='RGB',
        objects=['RGB','LMS'])

    image_space = param.ObjectSelector(
        default='XYZ', 
        objects=['XYZ']) # XYZ's all I've considered for now

    # CEBALERT: should be classselector
    display_space = param.Parameter(default=sRGB())
    display_sat = param.Number(default=1.0)
    display_val = param.Number(default=1.0)

    swap_polar_HSVorder = {
        'HSV': lambda HSV: HSV,
        'LCH': lambda LCH: numpy.dstack( (LCH[:,:,2],LCH[:,:,1],LCH[:,:,0]) )}
    

    def _convert(self,from_,to,what):
        fn = getattr(self.colorspace,'%s_to_%s'%(from_.lower(),to.lower())) # pretty hacky
        return fn(what)

    def analysis2receptors(self,a):
        a = self.swap_polar_HSVorder[self.analysis_space](a)        
        return self._convert(self.analysis_space,self.receptor_space,a)

    def receptors2analysis(self,r):
        a = self._convert(self.receptor_space,self.analysis_space,r)
        return self.swap_polar_HSVorder[self.analysis_space](a)
    
    def image2receptors(self,i):
        return self._convert(self.image_space,self.receptor_space,i)

    def analysis2display(self,a):
        a = self.swap_polar_HSVorder[self.analysis_space](a)
        fn = getattr(self.display_space,'%s_to_%s'%(self.analysis_space.lower(),'gammargb')) # pretty hacky
        return fn(a)

    # jitter, sat manips happen in analysis space
    
    def jitter_hue(self,a,amount):
        a = self.swap_polar_HSVorder[self.analysis_space](a)
        a[:,:,0] += amount
        a[:,:,0] %= 1.0

    def adjust_sat(self,a,factor):
        a = self.swap_polar_HSVorder[self.analysis_space](a)
        a[:,:,1] *= factor
        
