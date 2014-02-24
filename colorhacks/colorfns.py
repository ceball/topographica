from math import pi
import colorsys

import numpy

from topo.misc.inlinec import inline


# CEBALERT: rename these

## return Ma where M is 3x3 transformation matrix, for each pixel
def _threeDdot_dumb(M,a):
    result = numpy.empty(a.shape,dtype=a.dtype)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            A = numpy.array([a[i,j,0],a[i,j,1],a[i,j,2]]).reshape((3,1))
            L = numpy.dot(M,A)
            result[i,j,0] = L[0]
            result[i,j,1] = L[1]
            result[i,j,2] = L[2]

    return result

def _threeDdot_faster(M,a):
    swapped = a.swapaxes(0,2)
    shape = swapped.shape
    result = numpy.dot(M,swapped.reshape((3,-1)))
    result.shape = shape
    b = result.swapaxes(2,0)
    # need to do asarray to ensure dtype?
    return numpy.asarray(b,dtype=a.dtype)

# CB: probably could make a faster version if do aM instead,
# e.g. something like (never tested):
#def _threeDdot(M,a):
#    shape = a.shape
#    result = np.dot(a.reshape((-1,3)),M)
#    result.shape = shape
#    return result

threeDdot = _threeDdot_faster

def _abc_to_def_array(ABC,fn):
    shape = ABC[:,:,0].shape
    dtype = ABC.dtype
    
    DEF = numpy.zeros(shape,dtype=dtype)

    for i in range(shape[0]):
        for j in range(shape[1]):
            DEF[i,j,0],DEF[i,j,1],DEF[i,j,2]=fn(ABC[i,j,0],ABC[i,j,1],ABC[i,j,2])

    return DEF
    

def _rgb_to_hsv_array(RGB):
    """
    Equivalent to colorsys.rgb_to_hsv, except expects array like :,:,3
    """
    return _abc_to_def_array(RGB,colorsys.rgb_to_hsv)


def _hsv_to_rgb_array(HSV):
    """
    Equivalent to colorsys.hsv_to_rgb, except expects array like :,:,3
    """
    return _abc_to_def_array(HSV,colorsys.hsv_to_rgb)


# CEBALERT: should have opt in another file with automatic fallback,
# as in rest of topo

# CEBALERT: move lots of this stuff to a support file so it doesn't
# have to be read before the colorspace class.

def _rgb_to_hsv_array_opt(RGB):
    """Supposed to be equivalent to rgb_to_hsv_array()"""
    red = RGB[:,:,0]
    grn = RGB[:,:,1]
    blu = RGB[:,:,2]

    shape = red.shape
    dtype = red.dtype
    
    hue = numpy.zeros(shape,dtype=dtype)
    sat = numpy.zeros(shape,dtype=dtype)
    val = numpy.zeros(shape,dtype=dtype)

    code = """
//// MIN3,MAX3 macros from
// http://en.literateprograms.org/RGB_to_HSV_color_space_conversion_(C)
#define MIN3(x,y,z)  ((y) <= (z) ? \
                         ((x) <= (y) ? (x) : (y)) \
                     : \
                         ((x) <= (z) ? (x) : (z)))

#define MAX3(x,y,z)  ((y) >= (z) ? \
                         ((x) >= (y) ? (x) : (y)) \
                     : \
                         ((x) >= (z) ? (x) : (z)))
////

for (int i=0; i<Nred[0]; ++i) {
    for (int j=0; j<Nred[1]; ++j) {

        // translation of Python's colorsys.rgb_to_hsv()

        float r=RED2(i,j);
        float g=GRN2(i,j);
        float b=BLU2(i,j);

        float minc=MIN3(r,g,b); 
        float maxc=MAX3(r,g,b); 

        VAL2(i,j)=maxc;

        if(minc==maxc) {
            HUE2(i,j)=0.0;
            SAT2(i,j)=0.0;
        } else {
            float delta=maxc-minc; 
            SAT2(i,j)=delta/maxc;

            float rc=(maxc-r)/delta;
            float gc=(maxc-g)/delta;
            float bc=(maxc-b)/delta;

            if(r==maxc)
                HUE2(i,j)=bc-gc;
            else if(g==maxc)
                HUE2(i,j)=2.0+rc-bc;
            else
                HUE2(i,j)=4.0+gc-rc;

            HUE2(i,j)=(HUE2(i,j)/6.0);

            if(HUE2(i,j)<0)
                HUE2(i,j)+=1;
            //else if(HUE2(i,j)>1)
            //    HUE2(i,j)-=1;

        }

    }
}

"""
    inline(code, ['red','grn','blu','hue','sat','val'], local_dict=locals())

    return numpy.dstack((hue,sat,val))



def _hsv_to_rgb_array_opt(HSV):
    """Supposed to be equivalent to hsv_to_rgb_array()."""
    hue = HSV[:,:,0]
    sat = HSV[:,:,1]
    val = HSV[:,:,2]

    shape = hue.shape
    dtype = hue.dtype
    
    red = numpy.zeros(shape,dtype=dtype)
    grn = numpy.zeros(shape,dtype=dtype)
    blu = numpy.zeros(shape,dtype=dtype)

    code = """
for (int i=0; i<Nhue[0]; ++i) {
    for (int j=0; j<Nhue[1]; ++j) {

        // translation of Python's colorsys.hsv_to_rgb() using parts
        // of code from
        // http://www.cs.rit.edu/~ncs/color/t_convert.html
        float h=HUE2(i,j);
        float s=SAT2(i,j);
        float v=VAL2(i,j);

        float r,g,b;
        
        if(s==0) 
            r=g=b=v;
        else {
            int i=(int)floor(h*6.0);
            if(i<0) i=0;
            
            float f=(h*6.0)-i;
            float p=v*(1.0-s);
            float q=v*(1.0-s*f);
            float t=v*(1.0-s*(1-f));

            switch(i) {
                case 0:
                    r = v;
                    g = t;
                    b = p;
                    break;
                case 1:
                    r = q;
                    g = v;
                    b = p;
                    break;
                case 2:
                    r = p;
                    g = v;
                    b = t;
                    break;
                case 3:
                    r = p;
                    g = q;
                    b = v;
                    break;
                case 4:
                    r = t;
                    g = p;
                    b = v;
                    break;
                case 5:
                    r = v;
                    g = p;
                    b = q;
                    break;
            }
        }
        RED2(i,j)=r;
        GRN2(i,j)=g;
        BLU2(i,j)=b;
    }
}
"""
    inline(code, ['red','grn','blu','hue','sat','val'], local_dict=locals())
    return numpy.dstack((red,grn,blu))


rgb_to_hsv = _rgb_to_hsv_array_opt
hsv_to_rgb = _hsv_to_rgb_array_opt


KAP = 24389/27.0
EPS = 216/24389.0


def xyz_to_lab(XYZ,wp):

    X,Y,Z = numpy.dsplit(XYZ,3)
    xn,yn,zn = X/wp[0], Y/wp[1], Z/wp[2]

    def f(t):
        t = t.copy() # probably unnecessary! 
        t_eps = t>EPS
        t_not_eps = t<=EPS
        t[t_eps] = numpy.power(t[t_eps], 1.0/3)
        t[t_not_eps] = (KAP*t[t_not_eps]+16.0)/116.
        return t
            
    fx,fy,fz = f(xn), f(yn), f(zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)

    return numpy.dstack((L,a,b))


def lab_to_xyz(LAB,wp):

    L,a,b = numpy.dsplit(LAB,3)
    fy = (L+16)/116.0
    fz = fy - b / 200.0
    fx = a/500.0 + fy

    def finv(y):
        y =copy.copy(y) # CEBALERT: why copy?
        eps3 = EPS**3 
        return numpy.where(y > eps3,
                           numpy.power(y,3),
                           (116*y-16)/KAP)

    xr, yr, zr = finv(fx), finv(fy), finv(fz)
    return numpy.dstack((xr*wp[0],yr*wp[1],zr*wp[2]))

# CEBALERT: need to deal with LAB and LCH scales!! (maybe only LCH?)

def lch_to_lab(LCH):
    L,C,H = numpy.dsplit(LCH,3)
    return numpy.dstack( (L,C*numpy.cos(H),C*numpy.sin(H)) )

def lab_to_lch(LAB):
    L,A,B = numpy.dsplit(LAB,3)
    return numpy.dstack( (L, numpy.hypot(A,B), wrap(0,2*pi,numpy.arctan2(B,A))) )

