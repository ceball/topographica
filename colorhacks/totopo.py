import numpy
import param

import topo
from topo.base.arrayutil import clip_upper,wrap
from topo.sheet.basic import GeneratorSheet
from topo.pattern.image import FileImage,ImageSampler,PatternSampler
from topo.misc.commandline import global_params as GP
from topo import numbergen
from topo import pattern


def npyopen1(filename):
    rgbimage = numpy.load(filename)
    scaled = rgbimage/rgbimage.max()
    r = scaled[:,:,0]
    g = scaled[:,:,1]
    b = scaled[:,:,2]
    image = (r+g+b)/3.0 # pfff
    return image,r,g,b



# CB: :(
class ListGenerator(param.Parameterized):
    """
    When called, returns a list of the results of calling all its
    generators.
    """
    generators = param.List(default=[],doc="""
        List of callables used to produce the ListGenerator's items.""")

    def __call__(self):
        a = [g() for g in self.generators]
        return a


class HackedSelector(pattern.Selector):
    def get_current_generator(self):
        # only use inspect
        ind = self.inspect_value('index') 
        if ind is None:
            int_index=0
        else:
            int_index=int(len(self.generators)*wrap(0,1.0,self.inspect_value('index')))
        return self.generators[int_index]

    # hacked support color 
    def _get_red(self):
        return self.get_current_generator().red
    def _get_green(self):
        return self.get_current_generator().green
    def _get_blue(self):
        return self.get_current_generator().blue

    red = property(_get_red)
    green = property(_get_green)
    blue = property(_get_blue)



# CEBALERT: This class mutated to support all kinds of things in my
# analysis code. ColorImageSheet, ExtendToRGB, and the
# color-supporting FileImage subclasses are all a mess.
class ExtendToRGB(pattern.PatternGenerator):
    """
    Wrapper for any PatternGenerator to support red, green, and blue
    channels, e.g. for use with ColorImageSheet.

    If the specified generator itself has a 'generator' attribute,
    ExtendToRGB will attempt to get red, green, and blue from
    generator.generator (e.g. ColorImage inside a Selector);
    otherwise, ExtendToRGB will attempt to get red, green, and blue
    from generator. If no red, green, and blue are found in these
    ways, ExtendToRGB will synthesize the red, green, and blue
    channels.

    After finding or synthesizing red, green, and blue, they are
    scaled according to relative_channel_strengths.
    """
    # mostly these are hardcoded in various forms, so this is not useful.
    channels = ["red","green","blue"]

    generator = param.ClassSelector(class_=pattern.PatternGenerator,
                                    default=pattern.Constant())

    channel_factors = param.Dynamic(default=[1.,1.,1])

    hack_rg_grating = param.Boolean(default=False)

    correlate = param.Parameter(default=None)


    def __init__(self,**params):
        super(ExtendToRGB,self).__init__(**params)
        for c in self.channels:
            setattr(self,c,None)

    def hack_hook1(self):
        pass

    def hack_hook0(self,generator):
        pass

    def set_channel_values(self,p,params,gray,generator):
        channel_values = []
        # if the generator has the channels, take those values -
        # otherwise use gray*channel factors
        try:
            for chan,chan_strength in zip(self.channels,p.channel_factors):
                channel_values.append(getattr(generator,chan)*chan_strength)
        except AttributeError:
            for chan,chan_strength in zip(self.channels,p.channel_factors):            
                channel_values.append(gray*chan_strength)                    
        for nam,c in zip(self.channels,channel_values):                
            setattr(self,nam,c)
        

    def __call__(self,**params):
        p = param.ParamOverrides(self,params)

        ########
        # as for Selector etc, hack pass through certain parameters to
        # generator
        params['xdensity']=p.xdensity
        params['ydensity']=p.ydensity
        params['bounds']=p.bounds
        ########

        # (not **p)
        gray = p.generator(**params)

        #### GET GENERATOR ####
        # Got to get the generator that's actually making the pattern
        #
        # CEB: very hacky. maybe if
        # the various selector pattern generators had a way of
        # accessing the current generator's parameters, it could be
        # simpler?        
        if hasattr(p.generator,'get_current_generator'):
            # access the generator without causing any index to be advanced
            generator = p.generator.get_current_generator()
        elif hasattr(p.generator,'generator'):
            generator = p.generator.generator
        else:
            generator = p.generator
        #######################

        # CEBALERT: used to support non color patterns, too.        
        # (promoted red, green, blue from actual generator if it had
        # them, otherwise set to something based on gray)

        self.hack_hook0(generator)


        self.set_channel_values(p,params,gray,generator)

        # hacked correlation support
        if self.correlate is not None:
            corr_to,corr_from,corr_amt = self.correlate
            setattr(
                self,
                corr_to,
                corr_amt*getattr(self,corr_from)+(1-corr_amt)*getattr(self,corr_to))


        self.hack_hook1()
        
        return gray


class ColorImageSheet(GeneratorSheet):
    """
    A GeneratorSheet that handles RGB images.

    Accepts either a single-channel or an RGB input_generator.  If the
    input_generator stores separate red, green, and blue patterns, it
    is used as-is; other (monochrome) PatternGenerators are first
    wrapped using ExtendToRGB to create the RGB patterns.

    When a pattern is generated, a monochrome version is sent out on
    the Activity port as usual for a GeneratorSheet, and red, green,
    and blue activities are sent out on the RedActivity,
    GreenActivity, and BlueActivity ports.  Thus this class can be used
    just like GeneratorSheet, but with optional color channels.
    """

    src_ports=['Activity','RedActivity','GreenActivity','BlueActivity']

    constant_mean_total_retina_output = param.Number(default=None)

    hacky_e2rgb_class = param.Parameter(ExtendToRGB)

    def __init__(self,**params):
        super(ColorImageSheet,self).__init__(**params)
        self.activity_red=self.activity.copy()
        self.activity_green=self.activity.copy()
        self.activity_blue=self.activity.copy()


    def set_input_generator(self,new_ig,push_existing=False):
        """Wrap new_ig in ExtendToRGB if necessary."""
        # CEBALERT: Why have logic for supporting non-RGB
        # patterns in ExtendToRGB and in ColorImageSheet? 
        if not hasattr(new_ig,'red'):
            new_ig = self.hacky_e2rgb_class(generator=new_ig)
            
        super(ColorImageSheet,self).set_input_generator(new_ig,push_existing=push_existing)

        
    def generate(self):
        """
        Works as in the superclass, but also generates RGB output and sends
        it out on the RedActivity, GreenActivity, and BlueActivity ports.
        """
        super(ColorImageSheet,self).generate(applyofs=False)
        
        self.activity_red[:]   = self.input_generator.red
        self.activity_green[:] = self.input_generator.green
        self.activity_blue[:]  = self.input_generator.blue
        
            
        # HACKATTACK: abuse of output_fns list to allow one OF to be
        # applied repeatedly to each channel, or one OF per channel!
        # Also note does not apply OF to activity!
        if self.apply_output_fns:
            if len(self.output_fns)==0:
                pass
            elif len(self.output_fns)==1:
                output_fn = self.output_fns[0]
                output_fn(self.activity_red)
                output_fn(self.activity_green)
                output_fn(self.activity_blue)
            elif len(self.output_fns)==3: 
                self.output_fns[0](self.activity_red)
                self.output_fns[1](self.activity_green)
                self.output_fns[2](self.activity_blue)
            elif len(self.output_fns)==6: 
                self.output_fns[0](self.activity_red)
                self.output_fns[1](self.activity_green)
                self.output_fns[2](self.activity_blue)
                self.output_fns[3](self.activity_red)
                self.output_fns[4](self.activity_green)
                self.output_fns[5](self.activity_blue)
            else:
                raise

        if self.constant_mean_total_retina_output is not None:
            M = (self.activity_red+self.activity_green+self.activity_blue).mean()/3.0
            if M>0:
                p = self.constant_mean_total_retina_output/M
                for a in (self.activity_red,self.activity_green,self.activity_blue):
                    a*=p
                    # CEBALERT: hidden away OF
                    clip_upper(a,1.0)
                
        self.send_output(src_port='RedActivity',  data=self.activity_red)
        self.send_output(src_port='GreenActivity',data=self.activity_green)
        self.send_output(src_port='BlueActivity', data=self.activity_blue)



class HackedPatternSampler(PatternSampler):
    # at some point was having trouble allowing this to be none
    background_value_fn = param.Parameter(None,readonly=True)
    cache_image = param.Parameter(False,readonly=True)
    whole_pattern_output_fns = param.Parameter([],readonly=True)


class BaseColorImage(FileImage):
    random_generator = param.Callable(
        default=numbergen.UniformRandom(lbound=0,ubound=1,seed=1048921))

    pattern_sampler = param.ClassSelector(class_=ImageSampler,
        default=HackedPatternSampler(size_normalization='fit_shortest'))

    imopen = param.Callable(default=npyopen1)


    def __init__(self,**params):
        super(BaseColorImage,self).__init__(**params)
        self.red=self.green=self.blue=None


    def _reduced_call(self,**params_to_override):
        # PatternGenerator.__call__ but skipping stuff we don't need
        # to repeat.  Might be better to make each of red,green,blue
        # actually be some form of reduced patterngenerator and appear
        # in a generators list
        p=param.ParamOverrides(self,params_to_override)

        # This is what's skipped:
        #self._setup_xy(p.bounds,p.xdensity,p.ydensity,p.x,p.y,p.orientation)
            
        fn_result = self.function(p)
        self._apply_mask(p,fn_result)
        result = p.scale*fn_result+p.offset

        assert len(p.output_fns)==0
                               
        return result

    # Could move all opening stuff to an open_fn so that it
    # can be swapped out for simply reading an array, or
    # whatever.
    def _get_image(self,p):
        if p.filename!=self.last_filename or self._image is None:
            self.last_filename=p.filename

            self._image,self._image_red,self._image_green,self._image_blue = p.imopen(p.filename)

        return self._image

    def _again(self,p,**params_to_override):
        orig_image = self._image

        for col in ('red','green','blue'):
            self._image = getattr(self,"_image_%s"%col)
            setattr(self,col,self._reduced_call(**params_to_override))

        self._image = orig_image        

    def injection(self,p,gray):
        pass

    def __call__(self,**params_to_override):

        p = param.ParamOverrides(self,params_to_override)
        
        # HACKALERT
        params_to_override['cache_image']=True
        gray = super(BaseColorImage,self).__call__(**params_to_override)
        
        self._again(p,**params_to_override)

        self.injection(p,gray)
                            
        if p.cache_image is False:
            self._image_red=self._image_green=self._image_blue=self._image=None

        return gray


def dsplit_3D_to_2Ds(ABC):
    A,B,C = numpy.dsplit(ABC,3)
    for a in (A,B,C):
        a.shape = a.shape[0:2]
    return A,B,C


class ChannsFromImage(BaseColorImage):

    sat = param.Number(default=1.0)

    _hack_recording = param.Parameter(default=None)

    def injection(self,p,gray):

        im2pg = topo.sim.cconv.image2receptors
        pg2analysis = topo.sim.cconv.receptors2analysis
        analysis2pg = topo.sim.cconv.analysis2receptors
        jitterfn = topo.sim.cconv.jitter_hue
        satfn = topo.sim.cconv.multiply_sat
        
        ####
        
        channs_in  = numpy.dstack((self.red,self.green,self.blue))
        channs_out = im2pg(channs_in)
        analysis_space = pg2analysis(channs_out)

        if self._hack_recording is not None:            
            self._hack_recording(self,channs=channs_in,extra=analysis_space)            

        jitterfn(analysis_space,self.random_generator())
        satfn(analysis_space,self.sat)
        
        channs_out = analysis2pg(analysis_space)
        self.red,self.green,self.blue = dsplit_3D_to_2Ds(channs_out)
        
