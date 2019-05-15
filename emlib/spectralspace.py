import math
from sndtrck.spectrum import Spectrum
import bpf4 as bpf
from emlib.lib import intersection, returns_tuple


class Camera(object):
    def __init__(self, x=0, z=0, alpha=90, scanrate=1, focusdist=1):
        """
        Notes on settings:

        * the amount of 'zooming' that the camera produces depends on:
            - the scanperiod. The longer the period, the greater the zoom
            - the aperture (alpha) angle. This determines de length of the
              visible-field. The wider the angle, the smaller the zoom.

        x: position in the x-axis (time)
        z: position in the z-axis (the y-axis is frequency)
        alpha: aperture angle
        scanrate: a scanrate of 2 scans two secods in 1 second
        focusdist: distance at which scanrate operates

        """
        self.x = x
        self.z = z
        self.alpha = alpha
        self.scanrate = scanrate
        self.focusdist = focusdist

    def zoomfactor(self, z=None):
        """
        The zoomfactor of the camera, which depends on both the aperture (alpha)
        and the scanperiod.

        This is here for documentation. In fact, the zoomfactor can 
        be directly calculated by:

        camera.z_where_field_dur_is(z.scanperiod)
        """
        if z is None:
            z = self.z + self.focusdist
        visiblefieldrange = self.field_at(z)
        visiblefield = visiblefieldrange[1] - visiblefieldrange[0]
        factor = self.scanperiod / visiblefield
        return factor

    @property 
    def scanperiod(self):
        """
        The scan period is the duration of a scan when focusing
        to a screen at a distance of cam.focusdist
        """
        field1 = self.field_at(self.z + self.focusdist)
        dur = field1[1] - field1[0]
        return dur / self.scanrate

    @scanperiod.setter
    def scanperiod(self, period):
        normalperiod = self.scanperiod
        self.scanrate *= normalperiod / period

    def __repr__(self):
        return "Cam z:{z} x:{x} alpha:{a} per:{per}".format(
            z=self.z, x=self.x, a=self.alpha, per=self.scanperiod)

    def field_at_focus(self):
        return self.field_at(self.z + self.focusdist)

    def field_at(self, z):
        """
        Returns the line x0-x1 representing the vision field at z.
        If a layer would be present at z, the time x0-x1 would fill the field
        """
        zdif = z - self.z
        xh = zdif * math.tan(math.radians(self.alpha/2.))
        x0 = self.x - xh
        x1 = self.x + xh
        return x0, x1

    def z_where_field_dur_is(self, dur):
        """
        Get the absolute z where the field-duration 
        corresponds to the given one
        """
        zrel = dur / (2 * math.tan(math.radians(self.alpha/2.)))
        return zrel + self.z

    def intersect_at(self, z, x0, x1):
        """
        Return the intersection the visible field with a line defined 
        between x0-x1 at z, or None if no intersection
        """
        f0, f1 = self.field_at(z)
        return intersection(f0, f1, x0, x1)

    def relative_position_at(self, z, x0, x1):
        """
        Return the positions (r0, r1) of x0 and x1 relative to the field of
        view at z, defined between 0-1

        The relative position is defined so that a pos=0.5 is at the middle
        of the field of view, a pos at 0 is at the left of the field of view.

        A negative position is outside view, to the left, a pos > 1 is 
        outside view, to the right 

        If z lies behind the camera, then the returned positions are
        inverted, so that r0 > r1
        """
        f0, f1 = self.field_at(z)
        r0 = (x0 - f0) / (f1 - f0)
        r1 = (x1 - f0) / (f1 - f0)
        return r0, r1

    def project_visible(self, z, x0, x1):
        """
        Returns v0, v1, p0, p1 where:

        v0, v1: the visible part of x0-x1
        p0, p1: the coordinates in the projection screen corresponding to v0, v1

        If x0-x1 is not visible, returns None
        """
        visiblepart = self.intersect_at(z, x0, x1)
        if visiblepart is None:
            return None
        r0, r1 = self.relative_position_at(z, visiblepart[0], visiblepart[1])
        p0 = r0 * self.scanperiod
        p1 = r1 * self.scanperiod
        return visiblepart[0], visiblepart[1], p0, p1


class Layer(object):
    def __init__(self, spectrum, z, eqcurve=None):
        self.spectrum = spectrum
        self.z = z
        self.eqcurve = eqcurve
        
    @property
    def x0(self):
        return self.spectrum.t0 

    @property
    def x1(self):
        return self.spectrum.t1

    def render(self, x0=None, x1=None):
        if x0 is None and x1 is None:
            return self.spectrum
        x0 = self.x0 if x0 is None else x0
        x1 = self.x1 if x1 is None else x1
        return self.spectrum.partials_between(x0, x1, crop=True)


class Space(object):
    def __init__(self, layers):
        self.layers = layers
        self.layers.sort(key=lambda layer: layer.z)

    def addlayer(self, layer):
        self.layers.append(layer)
        self.layers.sort(key=lambda layer: layer.z)


class Timeline(object):
    def __init__(self, x, z, angle, focus=1, maxdist=10, scanrate=None, scanperiod=None):
        """
        A Timeline describes the movement of a camera in time

        x, z: position of the camera (x is "time", z is depth). Can be a bpf
        angle: the aperture of the camera. Can be a bpf.
        focus: xxxx
        maxdist: the distance at which the dynamic is minimum
        scanrate: the rate at which the visible field is rendered
        scanperiod: you can alternative set the scanrate by setting the period. 
                    One of them has to be unset
        """
        self.x = bpf.asbpf(x)
        self.z = bpf.asbpf(z)
        self.angle = bpf.asbpf(angle)
        self.focus = bpf.asbpf(focus)
        self.maxdist = bpf.asbpf(maxdist)
        self.scanrate = scanrate
        self.scanperiod = scanperiod
        assert scanrate is None or scanperiod is None

    def get_camera(self, t):
        cam = Camera(x=self.x(t), z=self.z(t), alpha=self.angle(t))
        if self.scanrate is not None:
            cam.scanrate = self.scanrate(t)
        elif self.scanperiod is not None:
            cam.scanperiod = self.scanperiod(t)
        return cam
    
    def get_gaincurve(self, t):
        return make_gaincurve(self.maxdist(t), self.focus(t))

    def evalspace(self, space, t, simulate=False, callback=None, flatten=True):
        """
        If callback is given, it will be called with the sectrum of each
        layer, as the layer is evaluated. This can be usefull if, for instance,
        the layer has been shrunk and needs to be reanalyzed

        callback: a function (layers, info) -> Spectrum, called for each layer. 
        """
        cam = self.get_camera(t)
        curve = self.get_gaincurve(t)
        return evalspace(space, cam, curve, simulate=simulate, 
                         callback=callback, flatten=flatten)


def make_gaincurve(maxdist, focus=1, exp=1):
    """
    A gain-curve maps z to gain, so that for any given z 
    in relation to a camera z, a gain is calculated

    In the normal case, the bigger the distance `z`, the lower the gain.
    For negative z (an object behind the camera) it is still possible
    to define a positive gain

    maxdist: the distance at which the gain decays to 0
    focus: the distance at which the gain has its maximum
    exp: the shape of the gain curve
    """
    return bpf.halfcosm(0, 0, focus, 1, maxdist, 0, exp=exp)


@returns_tuple("spectrum reports spectra info")
def evalspace(space, camera, gaincurve, simulate=False, callback=None, 
              flatten=True):
    """
    space: a Space
    camera: a Camera
    gaincurve: a bpf (normally the result of calling make_gaincurve), 
               mapping distance to gain
    callback: a function (spectrum, info) -> Spectrum
              will be called for each layer
              info = {'layerindex': ..}
              To pass extra parameters to callback, use functools.partial
    """
    spectra = []
    reports = []
    layeridx = 0
    for i, layer in enumerate(space.layers):
        zdiff = layer.z - camera.z
        gain = gaincurve(zdiff)
        if gain <= 0:
            continue
        projection = camera.project_visible(layer.z, layer.x0, layer.x1)
        if projection is None:
            continue
        v0, v1, p0, p1 = projection
        if simulate:
            spectrum = None
        else:
            spectrum = layer.spectrum.partials_between(v0, v1, crop=True)
            spectrum = spectrum.fit_between(p0, p1).scaleamp(gain)
            if layer.eqcurve is not None:
                spectrum = spectrum.equalize(layer.eqcurve)
            if callback:
                info = {'layerindex': layeridx}
                spectrum = callback(spectrum, info)
                if spectrum is None:
                    continue
            if layer.eqcurve is not None:
                spectrum = spectrum.equalize(layer.eqcurve)
        spectra.append(spectrum)
        reports.append({
            'visiblerange': (v0, v1), 
            'projectedrange': (p0, p1), 
            'z': layer.z, 
            'spectrum':spectrum,
            'cam_z': camera.z,
            'cam_alpha': camera.alpha,
            'cam_focusdist': camera.focusdist,
            'layer_index': i, 
            'gain': gain})
        layeridx += 1
        
    if flatten:
        allpartials = []
        for spectrum in spectra:
            allpartials.extend(spectrum.partials)
        flatspectrum = Spectrum(allpartials)
    else:
        flatspectrum = None
    info = {
        'camera.scanperiod': camera.scanperiod,
        'camera.focusdist': camera.focusdist,
    }
    return flatspectrum, reports, spectra, info
