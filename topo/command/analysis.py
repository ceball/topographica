"""
User-level analysis commands, typically for measuring or generating
SheetViews.

This file mostly consists of plotgroup declarations for existing
measurement commands defined in the FeatureMapper project. It also
implements several Topographica specific measurement commands,
including activity measurements (implemented by update_activity())
and weight matrix visualizations (e.g. update_projection).

The implementation of Activity plots for instance consists of the
update_activity() command plus the Activity PlotGroupTemplate.  The
update_activity() command reads the activity array of each Sheet and
makes a corresponding SheetView to put in the Sheet's sheet_views
dictionary, while the Activity PlotGroupTemplate specifies which
SheetViews should be plotted in which combination.  See the help for
PlotGroupTemplate for more information.
"""

import copy
import time
import sys

import numpy as np

import param
from param import ParameterizedFunction, ParamOverrides

from dataviews import SheetView, SheetStack, SheetLines, CoordinateGrid

from featuremapper.command import * # pyflakes:ignore (API import)

import topo
from topo.base.cf import CFSheet, Projection
from topo.base.sheet import Sheet
from topo.base.arrayutil import centroid
from topo.misc.attrdict import AttrDict
from topo.base.cf import CFProjection
from topo.analysis.featureresponses import pattern_present, pattern_response, update_activity  # pyflakes:ignore (API import)


class Collector(param.Parameterized):
    """
    A Collector collects the results of measurements over time,
    allowing any measurement to be easily saved as an animation. The
    output of any measurement can be recorded as well as sheet
    activities, projection CFs or projection activities:

    c = Collector()
    with r(100, steps=10):
       c.collect(topo.sim.V1)      # c.V1.Activity
       c.collect(measure_or_pref)  # c.V1.OrientationPreference etc.
       c.collect(measure_cog)      # c.V1.Afferent.CoG
       c.collect(topo.sim.V1.Afferent)  # c.V1.Afferent.CFGrid
       c.collect(topo.sim.V1.Afferent,  # c.V1.Afferent.Activity
                            activity=True)

   Once completed, the data may be easy obtained via attribute access
   as shown in the comments above. You may also pass keywords to the
   measurement via the record method or the sampling density (rows,
   cols) when recording CFs from projections.
    """


    measurements = param.List(default=[], doc="""
        A list of tuples of form (obj, kwargs) where obj may be a
        measurement class, a sheet or a projection.""")

    class group(object):
        """
        Container class for convenient attribute access.
        """
        def __repr__(self):
            return "Keys:\n   %s" % "\n   ".join(self.__dict__.keys())

    def __init__(self, **kwargs):
        super(Collector,self).__init__(**kwargs)

    def _projection_CFs(self, projection, **kwargs):
        """
        Record the CFs of a projection as a ProjectionGrid.
        """
        data = measure_projection.instance(projection=projection, **kwargs)()
        sheet = data.metadata['proj_dest_name']
        projection = data.metadata['info']
        projection_CFs = {}
        projection_CFs[sheet] = {}
        projection_CFs[sheet][projection] = {}
        projection_CFs[sheet][projection]['CFGrid'] = data
        return projection_CFs

    def _projection_activity(self, projection, **kwargs):
        """
        Record the projection activity of a projection.
        """
        sview = projection.projection_view()
        sheet = sview.metadata['src_name']
        stack = SheetStack(title=projection.name, bounds=sview.bounds,
                           initial_items=[(topo.sim.time(), sview)],
                           dimension_labels=['Time'],
                           time_type=[topo.sim.time.time_type])
        proj_name = sview.metadata['proj_name']
        projection_activity = {}
        projection_activity[sheet] = {}
        projection_activity[sheet][proj_name] = {}
        projection_activity[sheet][proj_name]['Activity'] = stack
        return projection_activity

    def _projection_measurement(self, projection, activity=False, **kwargs):
        """
        Record the CFs of a projection as a ProjectionGrid or record
        the projection activity SheetView.
        """
        if activity:
            return self._projection_activity(projection, **kwargs)
        else:
            return self._projection_CFs(projection, **kwargs)

    def _sheet_activity(self, sheet):
        """
        Given a sheet, return the data as a measurement in the
        appropriate dictionary format.
        """
        sview = sheet[:]
        stack = SheetStack(title=sview.name, bounds=sview.bounds,
                          initial_items=[(1.0,sview)], key_type=[float]) # topo.sim.time()
        activity_data = {}
        activity_data[sheet.name] = {}
        activity_data[sheet.name]['Activity'] =  stack
        return activity_data

    def _get_measurement(self, item):
        """
        Method to declare the measurements to collect views
        from. Accepts measurement classes or instances, sheet
        objects(to measure sheet activity) and projection objects (to
        measure CFs or projection activities).
        """
        if isinstance(item, tuple):
            obj, kwargs = item
        else:
            obj, kwargs = item, {}

        if isinstance(obj, CFProjection):
            return lambda :self._projection_measurement(obj, **kwargs)
        elif isinstance(obj, Sheet):
            return lambda: self._sheet_activity(obj)
        else:
            return obj.instance(**kwargs)

    def collect(self, obj, **kwargs):

        items = [self._formatter(el) for el in self.measurements]
        if kwargs:
            fmt = self._formatter((obj, kwargs))
            if fmt in items:
                raise Exception("%r already being recorded." % fmt)
            self.measurements.append((obj, kwargs))
        else:
            fmt = self._formatter(obj)
            if fmt in items:
                raise Exception("%r already being recorded." % fmt)
            self.measurements.append(obj)

    def save_entry(self, sheet_name, feature_name, data, projection=None):
        """
        Create or update entries, dynamically creating attributes for
        convenient access as necessary.
        """
        if not hasattr(self, sheet_name):
            setattr(self, sheet_name, self.group())
        group = getattr(self, sheet_name)

        if projection is not None:
            if not hasattr(group, projection):
                setattr(group, projection, self.group())
            group = getattr(group, projection)

        time_type = param.Dynamic.time_fn

        if 'Time' not in data.dimension_labels and not isinstance(data, CoordinateGrid):
            timestamped_data = data.add_dimension('Time', 0,
                                                  topo.sim.time(),
                                                  {'Time':{'type':time_type}})
        else:
            timestamped_data = data

        if not hasattr(group, feature_name):
            setattr(group, feature_name, timestamped_data)
        else:
            getattr(group, feature_name).update(timestamped_data)

    def _record_data(self, measurement_data):
        """
        Given measurement data in the standard dictionary format
        record the elements. The dictionary may be indexed by sheet
        name then by feature name (e.g. sheet measurements) or by
        sheet name, then projection name before the feature name
        (projection measurements).
        """
        for sheet_name in measurement_data:
            for name in measurement_data[sheet_name]:
                data = measurement_data[sheet_name][name]
                # Data may be a feature, or dictionary of projection labels
                if not isinstance(data, dict):
                    # Indexed by sheet name and feature name
                    self.save_entry(sheet_name, name, data, None)
                else:
                    # Indexed by sheet and projection name before the feature.
                    for feature_name in data:
                        self.save_entry(sheet_name, feature_name,
                                         data[feature_name], name)

    def run(self, durations, cycles=1):
        try:
            self.durations = list(durations) * cycles
        except:
            self.durations = [durations] * cycles
        return self


    def __enter__(self):
        self._old_measurements = self.measurements
        self.measurements = []
        return self

    def __exit__(self, exc, *args):
        self.advance(self.durations)
        self.measurements = self._old_measurements


    def advance(self, durations):
        measurements = [self._get_measurement(item) for item in self.measurements]
        for i,duration in enumerate(durations):
            try:
                # clear_output is needed to avoid displaying garbage
                # in static HTML notebooks.
                from IPython.core.display import clear_output
                clear_output()
            except:
                pass
            for measurement in measurements:
                measurement_data = measurement()
                self._record_data(copy.deepcopy(measurement_data))
            info = (i+1, len(durations), topo.sim.time())
            msg = "%d/%d measurement cycles complete. Simulation Time=%s" % info
            print '\r', msg
            sys.stdout.flush()
            time.sleep(0.0001)
            topo.sim.run(duration)
        print "Completed collection. Simulation Time=%s" % topo.sim.time()

    def _formatter(self, item):
        if isinstance(item, tuple):
            (obj, kwargs) = item
            return '(%s, %s)' % (obj.name, kwargs)
        else:
            return item.name

    def __repr__(self):

        if self.measurements == []:
            return 'Collector()'
        items = [self._formatter(el) for el in self.measurements]
        return 'Collector([%s])' % ', '.join(items)




class ProjectionSheetMeasurementCommand(param.ParameterizedFunction):
    """A callable Parameterized command for measuring or plotting a specified Sheet."""

    outputs = param.List(default=[],doc="""
        List of sheets to use in measurements.""")

    __abstract = True



class UnitMeasurementCommand(ProjectionSheetMeasurementCommand):
    """A callable Parameterized command for measuring or plotting specified units from a Sheet."""

    coords = param.List(default=[(0,0)],doc="""
        List of coordinates of unit(s) to measure.""")

    projection = param.ObjectSelector(default=None,doc="""
        Name of the projection to measure; None means all projections.""")

    __abstract = True

    def __call__(self,**params):
        p=ParamOverrides(self,params)
        for output in p.outputs:
            s = getattr(topo.sim,output,None)
            if s is not None:
                for x,y in p.coords:
                    s.update_unit_view(x,y,'' if p.projection is None else p.projection.name)


def update_rgb_activities():
    """
    Make available Red, Green, and Blue activity matrices for all appropriate sheets.
    """
    for sheet in topo.sim.objects(Sheet).values():
        metadata = AttrDict(src_name=sheet.name, precedence=sheet.precedence,
                            row_precedence=sheet.row_precedence,
                            timestamp=topo.sim.time())
        for c in ['Red','Green','Blue']:
            # should this ensure all of r,g,b are present?
            if hasattr(sheet,'activity_%s'%c.lower()):
                activity_copy = getattr(sheet,'activity_%s'%c.lower()).copy()
                new_view = SheetView(activity_copy, bounds=sheet.bounds, metadata=metadata)
                sheet.views.maps['%sActivity'%c]=new_view



class update_connectionfields(UnitMeasurementCommand):
    """A callable Parameterized command for measuring or plotting a unit from a Projection."""

    # Force plotting of all CFs, not just one Projection
    projection = param.ObjectSelector(default=None,constant=True)



class update_projection(UnitMeasurementCommand):
    """A callable Parameterized command for measuring or plotting units from a Projection."""



class measure_projection(param.ParameterizedFunction):

    rows = param.Number(default=10, doc="Number of CF rows.")

    cols = param.Number(default=10, doc="Number of CF columns.")

    projection = param.ObjectSelector(default=None, constant=True)

    def __call__(self, **params):
        p = ParamOverrides(self, params)
        return p.projection.grid(p.rows, p.cols)



class update_projectionactivity(ProjectionSheetMeasurementCommand):
    """
    Add SheetViews for all of the Projections of the ProjectionSheet
    specified by the sheet parameter, for use in template-based plots.
    """

    def __call__(self, **params):
        p = ParamOverrides(self, params)
        for sheet_name in p.outputs:
            s = getattr(topo.sim, sheet_name, None)
            if s is not None:
                for conn in s.in_connections:
                    if not isinstance(conn,Projection):
                        topo.sim.debug("Skipping non-Projection "+conn.name)
                    else:
                        v = conn.projection_view(topo.sim.time())
                        key = v.metadata.proj_name + 'ProjectionActivity'
                        topo.sim[v.metadata.src_name].views.maps[key] = v



class measure_cog(ParameterizedFunction):
    """
    Calculate center of gravity (CoG) for each CF of each unit in each CFSheet.

    Unlike measure_position_pref and other measure commands, this one
    does not work by collating the responses to a set of input patterns.
    Instead, the CoG is calculated directly from each set of incoming
    weights.  The CoG value thus is an indirect estimate of what
    patterns the neuron will prefer, but is not limited by the finite
    number of test patterns as the other measure commands are.

    Measures only one projection for each sheet, as specified by the
    proj_name parameter.  The default proj_name of '' selects the
    first non-self connection, which is usually useful to examine for
    simple feedforward networks, but will not necessarily be useful in
    other cases.
    """

    proj_name = param.String(default='',doc="""
        Name of the projection to measure; the empty string means 'the first
        non-self connection available'.""")

    stride = param.Integer(default=1, doc="Stride by which to skip grid lines"
                                          "in the CoG Wireframe.")

    def __call__(self, **params):
        p = ParamOverrides(self, params)

        measured_sheets = [s for s in topo.sim.objects(CFSheet).values()
                           if hasattr(s,'measure_maps') and s.measure_maps]

        results = {}

        # Could easily be extended to measure CoG of all projections
        # and e.g. register them using different names (e.g. "Afferent
        # XCoG"), but then it's not clear how the PlotGroup would be
        # able to find them automatically (as it currently supports
        # only a fixed-named plot).
        requested_proj=p.proj_name
        for sheet in measured_sheets:
            if sheet not in results:
                results[sheet.name] = {}
            for proj in sheet.in_connections:
                if (proj.name == requested_proj) or \
                   (requested_proj == '' and (proj.src != sheet)):
                   results[sheet.name][proj.name] = self._update_proj_cog(p, proj)

        return results


    def _update_proj_cog(self, p, proj):
        """Measure the CoG of the specified projection and register corresponding SheetViews."""

        sheet = proj.dest
        rows, cols = sheet.activity.shape
        xcog = np.zeros((rows, cols), np.float64)
        ycog = np.zeros((rows, cols), np.float64)

        for r in xrange(rows):
            for c in xrange(cols):
                cf = proj.cfs[r, c]
                r1, r2, c1, c2 = cf.input_sheet_slice
                row_centroid, col_centroid = centroid(cf.weights)
                xcentroid, ycentroid = proj.src.matrix2sheet(
                    r1 + row_centroid + 0.5,
                    c1 + col_centroid + 0.5)

                xcog[r][c] = xcentroid
                ycog[r][c] = ycentroid

        metadata = dict(precedence=sheet.precedence, row_precedence=sheet.row_precedence,
                        src_name=sheet.name, dimension_labels=['Time'], key_type=[topo.sim.time.time_type])

        timestamp = topo.sim.time()
        xsv = SheetView(xcog, sheet.bounds)
        ysv = SheetView(ycog, sheet.bounds)

        lines = []
        hlines, vlines = xsv.data.shape
        for hind in range(hlines)[::p.stride]:
            lines.append(np.vstack([xsv.data[hind,:].T, ysv.data[hind,:]]).T)
        for vind in range(vlines)[::p.stride]:
            lines.append(np.vstack([xsv.data[:,vind].T, ysv.data[:,vind]]).T)

        xcog_stack = SheetStack((timestamp, xsv), **metadata)
        ycog_stack = SheetStack((timestamp, ysv), **metadata)
        contour_stack = SheetStack((timestamp, SheetLines(lines, sheet.bounds)), **metadata)

        if 'XCoG' in sheet.views.maps:
            sheet.views.maps['XCoG'].update(xcog_stack)
        else:
            sheet.views.maps['XCoG'] = xcog_stack

        if 'YCoG' in sheet.views.maps:
            sheet.views.maps['YCoG'].update(ycog_stack)
        else:
            sheet.views.maps['YCoG'] = ycog_stack

        if 'CoG' in sheet.views.maps:
            sheet.views.maps['CoG'].update(contour_stack)
        else:
            sheet.views.maps['CoG'] = contour_stack

        return {'XCoG': xcog_stack, 'YCoG': ycog_stack, 'CoG': contour_stack}



from topo.plotting.plotgroup import create_plotgroup, save_plotgroup
from topo.command import pylabplot
from imagen.random import UniformRandom
from featuremapper import distribution


pg = create_plotgroup(name='Activity', category='Basic',
                      doc='Plot the activity for all Sheets.',
                      auto_refresh=True, pre_plot_hooks=[update_activity],
                      plot_immediately=True)
pg.add_plot('Activity', [('Strength', '_activity_buffer')])


pg = create_plotgroup(name='Connection Fields', category="Basic",
                     doc='Plot the weight strength in each ConnectionField of a specific unit of a Sheet.',
                     pre_plot_hooks=[update_connectionfields],
                     plot_immediately=True, normalize='Individually', situate=True)
pg.add_plot('Connection Fields', [('Strength', 'Weights')])


pg = create_plotgroup(name='Projection', category="Basic",
           doc='Plot the weights of an array of ConnectionFields in a Projection.',
           pre_plot_hooks=[update_projection],
           plot_immediately=False, normalize='Individually', sheet_coords=True)
pg.add_plot('Projection', [('Strength', 'Weights')])


pg = create_plotgroup(name='RGB', category='Other',
             doc='Combine and plot the red, green, and blue activity for all appropriate Sheets.',
             auto_refresh=True, pre_plot_hooks=[update_rgb_activities],
             plot_immediately=True)
pg.add_plot('RGB', [('Red', 'RedActivity'), ('Green', 'GreenActivity'),
                    ('Blue', 'BlueActivity')])


pg = create_plotgroup(name='Projection Activity', category="Basic",
                      doc='Plot the activity in each Projection that connects '
                          'to a Sheet.',
                      pre_plot_hooks=[update_projectionactivity.instance()],
                      plot_immediately=True, normalize='Individually',
                      auto_refresh=True)
pg.add_plot('Projection Activity', [('Strength', 'ProjectionActivity')])


pg= create_plotgroup(name='Center of Gravity',category="Preference Maps",
             doc='Measure the center of gravity of each ConnectionField in a Projection.',
             pre_plot_hooks=[measure_cog.instance()],
             plot_hooks=[pylabplot.topographic_grid.instance(xsheet_view_name="XCoG",ysheet_view_name="YCoG")],
             normalize='Individually')
pg.add_plot('X CoG',[('Strength','XCoG')])
pg.add_plot('Y CoG',[('Strength','YCoG')])
pg.add_plot('CoG',[('Red','XCoG'),('Green','YCoG')])


pg = create_plotgroup(name='RF Projection', category='Other',
                      doc='Measure white noise receptive fields.',
                      pre_plot_hooks=[measure_rfs.instance(
                          pattern_generator=UniformRandom())],
                      normalize='Individually')

pg.add_plot('RFs', [('Strength', 'RFs')])


pg = create_plotgroup(name='Orientation Preference', category="Preference Maps",
                      doc='Measure preference for sine grating orientation.',
                      pre_plot_hooks=[measure_sine_pref.instance(
                          preference_fn=distribution.DSF_WeightedAverage())])
pg.add_plot('Orientation Preference', [('Hue', 'OrientationPreference')])
pg.add_plot('Orientation Preference&Selectivity',
            [('Hue', 'OrientationPreference'),
             ('Confidence', 'OrientationSelectivity')])
pg.add_plot('Orientation Selectivity', [('Strength', 'OrientationSelectivity')])
pg.add_plot('Phase Preference', [('Hue', 'PhasePreference')])
pg.add_plot('Phase Selectivity', [('Strength', 'PhaseSelectivity')])
pg.add_static_image('Color Key', 'static/or_key_white_vert_small.png')


pg = create_plotgroup(name='vonMises Orientation Preference',
                      category="Preference Maps",
                      doc='Measure preference for sine grating orientation '
                          'using von Mises fit.',
                      pre_plot_hooks=[measure_sine_pref.instance(
                          preference_fn=distribution.DSF_VonMisesFit(),
                          num_orientation=16)])
pg.add_plot('Orientation Preference', [('Hue', 'OrientationPreference')])
pg.add_plot('Orientation Preference&Selectivity',
            [('Hue', 'OrientationPreference'),
             ('Confidence', 'OrientationSelectivity')])
pg.add_plot('Orientation Selectivity', [('Strength', 'OrientationSelectivity')])
pg.add_plot('Phase Preference', [('Hue', 'PhasePreference')])
pg.add_plot('Phase Selectivity', [('Strength', 'PhaseSelectivity')])
pg.add_static_image('Color Key', 'static/or_key_white_vert_small.png')


pg = create_plotgroup(name='Bimodal Orientation Preference',
                      category="Preference Maps",
                      doc='Measure preference for sine grating orientation '
                          'using bimodal von Mises fit.',
                      pre_plot_hooks=[measure_sine_pref.instance(
                          preference_fn=distribution.DSF_BimodalVonMisesFit(),
                          num_orientation=16)])
pg.add_plot('Orientation Preference', [('Hue', 'OrientationPreference')])
pg.add_plot('Orientation Preference&Selectivity',
            [('Hue', 'OrientationPreference'),
             ('Confidence', 'OrientationSelectivity')])
pg.add_plot('Orientation Selectivity', [('Strength', 'OrientationSelectivity')])
pg.add_plot('Second Orientation Preference',
            [('Hue', 'OrientationMode2Preference')])
pg.add_plot('Second Orientation Preference&Selectivity',
            [('Hue', 'OrientationMode2Preference'),
             ('Confidence', 'OrientationMode2Selectivity')])
pg.add_plot('Second Orientation Selectivity',
            [('Strength', 'OrientationMode2Selectivity')])
pg.add_static_image('Color Key', 'static/or_key_white_vert_small.png')


pg = create_plotgroup(name='Two Orientation Preferences',
                      category='Preference Maps',
                      doc='Display the two most preferred orientations for '
                          'each units, using bimodal von Mises fit.',
                      pre_plot_hooks=[measure_sine_pref.instance(
                          preference_fn=distribution.DSF_BimodalVonMisesFit(),
                          num_orientation=16)])
pg.add_plot('Two Orientation Preferences', [('Or1', 'OrientationPreference'),
                                            ('Sel1', 'OrientationSelectivity'),
                                            ('Or2', 'OrientationMode2Preference'),
                                            ('Sel2', 'OrientationMode2Selectivity')])
pg.add_static_image('Color Key', 'static/two_or_key_vert.png')


pg = create_plotgroup(name='Spatial Frequency Preference',
                      category="Preference Maps",
                      doc='Measure preference for sine grating orientation '
                          'and frequency.',
                      pre_plot_hooks=[measure_sine_pref.instance(
                          preference_fn=distribution.DSF_WeightedAverage())])
pg.add_plot('Spatial Frequency Preference',
            [('Strength', 'FrequencyPreference')])
pg.add_plot('Spatial Frequency Selectivity',
            [('Strength', 'FrequencySelectivity')])
# Just calls measure_sine_pref to plot different maps.


pg = create_plotgroup(name='Ocular Preference', category="Preference Maps",
                      doc='Measure preference for sine gratings between two '
                          'eyes.',
                      pre_plot_hooks=[measure_od_pref.instance()])
pg.add_plot('Ocular Preference', [('Strength', 'OcularPreference')])
pg.add_plot('Ocular Selectivity', [('Strength', 'OcularSelectivity')])


pg= create_plotgroup(name='PhaseDisparity Preference',category="Preference Maps",doc="""
    Measure preference for sine gratings at a specific orentation differing in phase
    between two input sheets.""",
             pre_plot_hooks=[measure_phasedisparity.instance()],normalize='Individually')
pg.add_plot('PhaseDisparity Preference', [('Hue', 'PhasedisparityPreference')])
pg.add_plot('PhaseDisparity Preference&Selectivity',
            [('Hue', 'PhasedisparityPreference'),
             ('Confidence', 'PhasedisparitySelectivity')])
pg.add_plot('PhaseDisparity Selectivity',
            [('Strength', 'PhasedisparitySelectivity')])
pg.add_static_image('Color Key', 'static/disp_key_white_vert_small.png')


pg = create_plotgroup(name='Direction Preference', category="Preference Maps",
                      doc='Measure preference for sine grating movement '
                          'direction.',
                      pre_plot_hooks=[measure_dr_pref.instance()])
pg.add_plot('Direction Preference', [('Hue', 'DirectionPreference')])
pg.add_plot('Direction Preference&Selectivity', [('Hue', 'DirectionPreference'),
                                                 ('Confidence',
                                                  'DirectionSelectivity')])
pg.add_plot('Direction Selectivity', [('Strength', 'DirectionSelectivity')])
pg.add_plot('Speed Preference', [('Strength', 'SpeedPreference')])
pg.add_plot('Speed Selectivity', [('Strength', 'SpeedSelectivity')])
pg.add_static_image('Color Key', 'static/dr_key_white_vert_small.png')


pg = create_plotgroup(name='Hue Preference', category="Preference Maps",
                      doc='Measure preference for colors.',
                      pre_plot_hooks=[measure_hue_pref.instance()],
                      normalize='Individually')
pg.add_plot('Hue Preference', [('Hue', 'HuePreference')])
pg.add_plot('Hue Preference&Selectivity',
            [('Hue', 'HuePreference'), ('Confidence', 'HueSelectivity')])
pg.add_plot('Hue Selectivity', [('Strength', 'HueSelectivity')])


pg = create_plotgroup(name='Second Orientation Preference',
                      category="Preference Maps",
                      doc='Measure the second preference for sine grating '
                          'orientation.',
                      pre_plot_hooks=[
                          measure_second_or_pref.instance(true_peak=False)])
pg.add_plot('Second Orientation Preference',
            [('Hue', 'OrientationMode2Preference')])
pg.add_plot('Second Orientation Preference&Selectivity',
            [('Hue', 'OrientationMode2Preference'),
             ('Confidence', 'OrientationMode2Selectivity')])
pg.add_plot('Second Orientation Selectivity',
            [('Strength', 'OrientationMode2Selectivity')])
pg.add_static_image('Color Key', 'static/or_key_white_vert_small.png')


pg = create_plotgroup(name='Second Peak Orientation Preference',
                      category="Preference Maps",
                      doc='Measure the second peak preference for sine '
                          'grating orientation.',
                      pre_plot_hooks=[
                          measure_second_or_pref.instance(true_peak=True)])
pg.add_plot('Second Peak Orientation Preference',
            [('Hue', 'OrientationMode2Preference')])
pg.add_plot('Second Peak Orientation Preference&Selectivity',
            [('Hue', 'OrientationMode2Preference'),
             ('Confidence', 'OrientationMode2Selectivity')])
pg.add_plot('Second Peak Orientation Selectivity',
            [('Strength', 'OrientationMode2Selectivity')])
pg.add_static_image('Color Key', 'static/or_key_white_vert_small.png')


pg = create_plotgroup(name='Two Peaks Orientation Preferences',
                      category='Preference Maps',
                      doc="""Display the two most preferred orientations for
                      all units with a multimodal orientation preference
                      distribution.""",
                      pre_plot_hooks=[
                          measure_second_or_pref.instance(num_orientation=16,
                                                          true_peak=True)])
pg.add_plot('Two Peaks Orientation Preferences',
            [('Or1', 'OrientationPreference'),
             ('Sel1', 'OrientationSelectivity'),
             ('Or2', 'OrientationMode2Preference'),
             ('Sel2', 'OrientationMode2Selectivity')])
pg.add_static_image('Color Key', 'static/two_or_key_vert.png')


pg = create_plotgroup(name='Corner OR Preference', category="Preference Maps",
                      doc='Measure orientation preference for corner shape ('
                          'or other complex stimuli that cannot be '
                          'represented as fullfield patterns).',
                      pre_plot_hooks=[measure_corner_or_pref.instance(
                          preference_fn=distribution.DSF_WeightedAverage())],
                      normalize='Individually')
pg.add_plot('Corner Orientation Preference', [('Hue', 'OrientationPreference')])
pg.add_plot('Corner Orientation Preference&Selectivity',
            [('Hue', 'OrientationPreference'),
             ('Confidence', 'OrientationSelectivity')])
pg.add_plot('Corner Orientation Selectivity',
            [('Strength', 'OrientationSelectivity')])


pg = create_plotgroup(name='Corner Angle Preference',
                      category="Preference Maps",
                      doc='Measure preference for angles in corner shapes',
                      normalize='Individually')
pg.pre_plot_hooks = [measure_corner_angle_pref.instance()]
pg.add_plot('Corner Angle Preference', [('Hue', 'AnglePreference')])
pg.add_plot('Corner Angle Preference&Selectivity',
            [('Hue', 'AnglePreference'), ('Confidence', 'AngleSelectivity')])
pg.add_plot('Corner Angle Selectivity', [('Strength', 'AngleSelectivity')])
pg.add_plot('Corner Orientation Preference', [('Hue', 'OrientationPreference')])
pg.add_plot('Corner Orientation Preference&Selectivity',
            [('Hue', 'OrientationPreference'),
             ('Confidence', 'OrientationSelectivity')])
pg.add_plot('Corner Orientation Selectivity',
            [('Strength', 'OrientationSelectivity')])
pg.add_static_image('Hue Code', 'static/key_angles.png')


pg= create_plotgroup(name='Position Preference',category="Preference Maps",
           doc='Measure preference for the X and Y position of a Gaussian.',
           pre_plot_hooks=[measure_position_pref.instance(
            preference_fn=distribution.DSF_WeightedAverage(selectivity_scale=(0.,17.) ))],
           plot_hooks=[pylabplot.topographic_grid.instance()],
           normalize='Individually')

pg.add_plot('X Preference',[('Strength','XPreference')])
pg.add_plot('Y Preference',[('Strength','YPreference')])
pg.add_plot('Position Preference',[('Red','XPreference'),
                                   ('Green','YPreference')])


create_plotgroup(template_plot_type="curve",name='Orientation Tuning Fullfield',category="Tuning Curves",doc="""
            Plot orientation tuning curves for a specific unit, measured using full-field sine gratings.
            Although the data takes a long time to collect, once it is ready the plots
            are available immediately for any unit.""",
        pre_plot_hooks=[measure_or_tuning_fullfield.instance()],
        plot_hooks=[pylabplot.cyclic_tuning_curve.instance(x_axis='orientation')])


create_plotgroup(template_plot_type="curve",name='Orientation Tuning',category="Tuning Curves",doc="""
            Measure orientation tuning for a specific unit at different contrasts,
            using a pattern chosen to match the preferences of that unit.""",
        pre_plot_hooks=[measure_or_tuning.instance()],
        plot_hooks=[pylabplot.cyclic_tuning_curve.instance(x_axis="orientation")],
        prerequisites=['XPreference'])


create_plotgroup(template_plot_type="curve",name='Size Tuning',category="Tuning Curves",
        doc='Measure the size preference for a specific unit.',
        pre_plot_hooks=[measure_size_response.instance()],
        plot_hooks=[pylabplot.tuning_curve.instance(x_axis='size')],
        prerequisites=['OrientationPreference','XPreference'])


create_plotgroup(template_plot_type="curve",name='Contrast Response',category="Tuning Curves",
        doc='Measure the contrast response function for a specific unit.',
        pre_plot_hooks=[measure_contrast_response.instance()],
        plot_hooks=[pylabplot.tuning_curve.instance(x_axis="contrast")],
        prerequisites=['OrientationPreference','XPreference'])


create_plotgroup(template_plot_type="curve",name='Frequency Tuning',category="Tuning Curves",
        doc='Measure the spatial frequency preference for a specific unit.',
        pre_plot_hooks=[measure_frequency_response.instance()],
                 plot_hooks=[pylabplot.tuning_curve.instance(x_axis="frequency")],
        prerequisites=['OrientationPreference','XPreference'])


create_plotgroup(template_plot_type="curve",name='Orientation Contrast',category="Tuning Curves",
                 doc='Measure the response of one unit to a center and surround sine grating disk.',
                 pre_plot_hooks=[measure_orientation_contrast.instance()],
                 plot_hooks=[pylabplot.cyclic_tuning_curve.instance(x_axis="orientationsurround", center=False,
                                                                    relative_labels=True)],
                 prerequisites=['OrientationPreference','XPreference'])





import types

__all__ = list(set([k for k, v in locals().items()
                    if isinstance(v, types.FunctionType) or (isinstance(v, type)
                    and issubclass(v, ParameterizedFunction))
                    and not v.__name__.startswith('_')]))
