import __main__

import numpy

import param

from topo.misc.commandline import global_params as p

import colorhacks.colorspaces

p.add(
 
    #######################################
    # input data, color spaces, receptor types
    dataset=param.ObjectSelector(default='Barca',objects=['Barca']),    
    
    dataset_subset = param.Parameter(default='natural'),

    dataset_colorspace = param.String(default="XYZ"),

    receptor_responses = param.ObjectSelector(
        default = "sRGB",
        objects = ["sRGB","spLMS"]),

    dataset_format = param.ObjectSelector(
        default="npy",
        objects=["npy","tiff"]),

    analysis_colorspace = param.ObjectSelector(
        default="HSV",
        objects=["HSV","LCH"]),

    #######################################

    
    #######################################
    # NETWORK STRUCTURE
    blue = param.Boolean(default=True),
    redgreen = param.Boolean(default=True),    
    blueOFF = param.Boolean(default=True),
    BYOPP = param.Boolean(default=False), # blue-yellow opponent vs coextensive
    BYOPPhack = param.Boolean(default=False), # also having RedGreen-Blue
    lumpath = param.Boolean(default=True),
    LGN = param.Boolean(default=True),
    V1 = param.Boolean(default=True),
    A = param.Boolean(default=True), # haven't checked without A for a while
    # CEBALERT: unless you make it a dim, should be bool
    rgcsym = param.Number(default=1.0,bounds=(0.0,1.0)),
    censcale = param.Number(default=1.0),
    surscale = param.Number(default=1.0),
    rgc_upper_limit = param.Number(default=None,allow_None=True),

    #######################################

    
    ###############################################################
    # CEBALERT:
    e2rgb_divide_channels_by_max_of_channels = param.Boolean(default=True),
    ###############################################################


    #######################################
    # normalization
    constant_mean_total_retina_output = param.Number(default=0.44,allow_None=True),
    constant_mean_total_lgn_output = param.Number(default=0.1,doc="Desired (spatial) mean activity of one LGN unit (not mean over time!).",allow_None=True),
    LC = param.Number(default=-0.2,bounds=(-1.0,1.0)),
    #######################################

    hjitter = param.Number(default=1.0,bounds=(0.0,1.0)),

    correlation = param.Number(default=None,allow_None=True,bounds=(0.0,1.0)),
    correlate_what = param.List(default=["green","red"]),

    mrc_or_duration = param.Number(default=0.175),

    rh_seed = param.Number(default=1048921),
    porderseed = param.Number(default=2042),
    numpy_random_seed = param.Parameter((500,500)),
    input_seed = param.Number(500)
    )


########################################################
# stupid checks, setting one param based on another, etc
    
if p.BYOPP is True: 
    assert (p.rgcsym==0)

if not p.LGN:
    print 'no LGN; will not create V1'
    p.V1 = False

if not p.blueOFF:
    assert p.rgcsym==0
    assert not p.BYOPP
    assert not p.BYOPPhack

########################################################

if p.receptor_responses == 'sRGB':
    colorspace = colorhacks.colorspaces.sRGB()
    receptor_space = 'RGB'
elif p.receptor_responses == 'spLMS':
    colorspace = colorhacks.colorspaces.spLMS()
    receptor_space = 'LMS'
else:
    raise

cconv = colorhacks.colorspaces.TopoColorConverter(
    colorspace = colorspace,
    analysis_space = p.analysis_colorspace,
    image_space = p.dataset_colorspace,
    receptor_space = receptor_space)


