from __future__ import annotations
import math
from pitchtools import amp2db, db2amp
import re as _re


def voltage_divider(Vin, R1, R2):
    """
    return the voltage resulting of the voltage divider 
    represented by R1 and R2

    Args:
        R1: resistance in ohms
        R2: resistance in ohms

    Returns:
        the resulting voltage
    """
    R1, R2 = _normalizevalue(R1), _normalizevalue(R2)
    Vout = R2 / (R1+R2) * Vin
    return _Volt(Vout)


def voltage_divider_R2(Vin, Vout, R1="1K"):
    """
    calculate R2 of a voltage divider with a given R1
    """
    R1 = _normalizevalue(R1)
    R2 = R1*(1/(Vin/Vout - 1))
    return _Res(R2)


def voltage_divider_R1(Vin, Vout, R2="10K"):
    """
    calculate R1 of a voltage divider with a given R2"""
    R2 = _normalizevalue(R2)
    R1 = R2 / (1/(Vin/Vout - 1))
    return _Res(R1)


def opamp_amplifier_Rf(gain, Rgnd):
    """
    Calculate the feedback resistor of an opamp (non-inverting) for a given gain and Rgnd

    ::

        Vin ---------------[+]
                              >------+---- Vout = Vin*gain
                        +--[-]       |
                        |            Rf
                        +------------+
                                     |
                                     Rgnd
                                     |
                                    GND

    """
    Rgnd = _normalizevalue(Rgnd)
    Rf = Rgnd*(gain-1)
    return _Res(Rf)


def opamp_amplifier_Rgnd(gain, Rf):
    """
    Calculate the resistor to ground for the given gain and feedback resistor (non-inverting)

    See opamp_amplifier_Rf for schematic
    """
    Rf = _normalizevalue(Rf)
    Rgnd = Rf/(gain-1)
    return _Res(Rgnd)


def opamp_inverting_gain(R1, Rf):
    """
    Return the gain of the opamp for the given values of the resistors

                  --Rf---
                  |     |
    Vin----R1----[-]\   |
                     ------ Out
     Bias or Gnd-[+]/
    """
    R1, Rf = map(_normalizevalue, (R1, Rf))
    gain = -(Rf/R1)
    return gain


def opamp_gain(R1, Rf):
    """
        +-----------+                
Vin+----> +         |                
        |       OUT +---+---> Vout  
    +---> -         |   |            
    |   +-----------+   |            
    |                   |
    |                  Rf
    |                   |
    +-------------------+
                        |            
                       R1
                        |
                       GND
    """
    R1, Rf = map(_normalizevalue, (R1, Rf))
    gain = 1 + (Rf/R1)
    return gain


def differential_amplifier_Vout(Vin, Vref, R1, R2, R3, R4):
    """
    Return Vout for the given circuit

                ---- R4---
                |        |
    Vref -- R3----[-]    |
                     >---+--- Vout
    Vin --- R1----[+]
               |
               R2
               |
              GND

    Ref.: http://www.electronics-tutorials.ws/opamp/opamp_5.html
    """
    R1, R2, R3, R4 = map(_normalizevalue, (R1, R2, R3, R4))
    Vout = -Vin*(R3/R1) + Vref*(R4/(R2+R4)*((R1+R3)/R1))
    return _Volt(Vout)


def differential_amplifier_scale(Vin_range, Vout_range,
                                 Vref, R1="10K", R2="10K"):
    """
    Vin_range: (Vmin, Vmax) that this input will take
    Vout_range: (Vmin, Vmax) that the output should have

                ---- R2----GND 
                |         
    V1   -- R1----[+]\    
                      \______ Vout
                      /   |
    Vref -- R3----[-]/    |
               |          |
               R4---------|
               |
              GND

    Returns R2, R4
    """
    R1 = _normalizevalue(R1)
    R2 = _normalizevalue(R2)
    dVout = Vout_range[1] - Vout_range[0]
    dVin = Vin_range[1] - Vin_range[0]
    gain = dVout/dVin
    offset = Vout_range[0] - Vin_range[0] * gain
    R4 = -(offset/Vref * R2)
    x = 1 + (R4/R2)
    R3 = R1 / (x/gain - 1)
    return _Res(R3), _Res(R4)


def rcfilter_freq(R, C):
    """
    Example
    =======

    Low-pass:  Vin ---R--+---Vout
                         |
                         C--GND

    High-pass: Vin--C--+--Vout
                       |
                       R--GND

    Active Highpass
                          ----R2----
                          |        |
               Vin--C--R--+- [-] \ |
                                 -+--Vout
                          +- [+] /
                          |
                         GND

    """
    C = _normalizevalue(C)
    R = _normalizevalue(R)
    fc = 1 / (2*math.pi*R*C)
    return fc


def rcfilter_R(freq, C=0.1*1e-06):
    """find the value of R for the given cutoff freq."""
    C = _normalizevalue(C)
    R = ((1/freq)/C)/(2*math.pi)
    return _Res(R)


def rcfilter_C(freq, R):
    """return the value of C for the given cutoff freq and resistance"""
    R = _normalizevalue(R)
    C = 1/(R*freq*2*math.pi)
    return _Cap(C)


def _display_capacitance(C):
    """C: capacitance value in F"""
    pF = C*1e12
    if pF > 1000000000:
        return "%.2fmF" % (pF / 1000000000)
    elif pF > 1000000:
        return "%.2fuF" % (pF / 1000000)
    elif pF > 1000:
        return "%.2fnF" % (pF / 1000)
    return "%dpF" % int(pF)


def _display_resistance(R):
    if R > 1000000:
        return "%.2f megOhm" % (R/1000000)
    elif R > 1000:
        return "%.2f kOhm" % (R/1000)
    else:
        return "%s Ohm" % str(round(R, 2))


def _display_voltage(v):
    if 0 < v < 1:
        return "%s mV" % str(round(v*1000, 1))
    else:
        return str(v)


def _display_current(I):
    if I < 0.01:
        return "%s mA" % (str(round(I*1000, 2)))
    return "%s A" % str(I)


class _Cap(float):
    def __repr__(self):
        return _display_capacitance(self)


class _Res(float):
    def __repr__(self):
        return _display_resistance(self)


class _Volt(float):
    def __repl__(self):
        return _display_voltage(self)


class _Curr(float):
    def __repl__(self):
        return _display_current(self)


KOhm = K = 1000
uF = 1e-06
mV = 0.001


def gain2dB(gain):
    """
    Convert voltage gain to decibels

    gain: the ratio between two voltages V2/V1
    """
    dB = 20*math.log(gain)
    return dB


def dB2gain(dB):
    """
    Convert gain from dB to voltage ratio

    Example
    =======

    given two signals of voltage V1 and V2, an amplifier
    provides a gain of G measured in dB. 

    V2/V1 = db2gain(G)

    See also: gain2dB
    """
    V = math.exp(dB/20)
    return V


def parallel_resistors(R1, R2):
    """
    calculate the resulting resistance

    o------+------+
           |      |
           R1     R2
           |      |
    o------+------+

    (R1 and R2 are commutative)
    """
    Rtotal = (R1*R2) / (R1+R2)
    return _Res(Rtotal)


def parallel_capacitors(*capacitors):
    """
    capacitors: the capacitors to connect in parallel
    scale: a numerical value or a unit like 'u' or 'n'
    """
    caps = [_normalizevalue(cap) for cap in capacitors]
    return _Cap(sum(caps))


def series_capacitors(*capacitors):
    """
    1/Cr = 1/C1 + 1/C2 + ...
    """
    return _Cap(1/sum(1/_normalizevalue(cap) for cap in capacitors))
   

def _normalizevalue(value):
    """
    value can be:
        * a numeric value
        * a string of the sort '10V', '20K', '0.1uF'
        * a sequence of the above, in which case a sequence of the same
          type will be returned
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, str):
        splitpoint = _re.search("[a-zA-Z]", value)
        if not splitpoint:
            return float(value)
        num = float(value[:splitpoint.start()])
        unit = value[splitpoint.start():]
        return num * _getscaling(unit)           
    elif hasattr(value, '__iter__'):
        seq = value
        cls = seq.__class__
        values = cls(map(_normalizevalue, seq))
        return values


def _getscaling(unit):
    if isinstance(unit, (int, float)):
        return unit
    if unit[-1] in 'VFA':
        unit = unit[:-1]
    scalingvalue = {
        'u': 1e-06,
        'n': 1e-09,
        'k': 1000,
        'meg': 1000*1000,
        'p': 1e-12,
        'm': 0.001
        }.get(unit.lower())
    if scalingvalue is None:
        raise ValueError("unit not understood")
    return scalingvalue


def spice_pwm(freq, duty=0.5, voltage=5, phase=0):
    """
    Returns the spice directive and the parameters to generate
    a PWM signal.
    """
    Vinitial = 0
    Von = voltage
    Trise = Tfall = "0.1u"
    Tperiod = 1.0/freq
    Ton = Tperiod * duty
    Tdelay = (phase/(2*math.pi)) * Tperiod
    Ncycles = ""
    s = "PULSE({Vinitial} {Von} {Tdelay} {Trise} {Tfall}" \
        "{Ton} {Tperiod} {Ncycles})"
    s = s.format(**locals())
    out = {'Vinitial': Vinitial, 'Von': Von, 'Trise': Trise,
           'Tperiod': Tperiod, 'Ton': Ton, 'Tdelay': Tdelay,
           'Ncycles': Ncycles}
    return s, out


def ohmlaw(V=None, I=None, R=None):
    """
    provide two of the values, the third one
    is the result

    V = I/R
    """
    V, I, R = map(_normalizevalue, (V, I, R))
    if len([x for x in (V, I, R) if x is None]) != 1:
        raise ValueError("two values must be given")
    if V is None:
        V = I*R
        return _Volt(V)
    elif I is None:
        I = V/R
        return _Curr(I)
    else:
        R = V/I
        return _Res(R)


def capacitive_reactance(cap, freq):
    """
    In an AC Circuit the applied voltage signal is continually
    changing from a + to - . The capacitor is being charged or
    discharged on a continuous basis.
    
    As the capacitor charges or discharges, a current flows
    through it which is restricted by the internal resistance
    of the capacitor. This internal resistance is commonly known
    as Capacitive Reactance and is given the symbol Xc in Ohms.
    
    Unlike resistance which has a fixed value, Xc varies with
    the applied frequency so any variation in supply frequency.

    """
    cap = _normalizevalue(cap)
    Xc = 1/(2*math.pi*freq*cap)
    return _Res(Xc)


def microphone_sensitivity(transferfactor):
    """
    transferfactor in mV/Pa

    Returns the sensitivity in mV/Pa
    """
    return amp2db(transferfactor/1000.)


def microphone_transferfactor(sensitivity):
    """
    sensitivity in dB re 1V/Pa

    Returns the transfer-factor in mV/Pa
    """
    a = db2amp(sensitivity)
    return a * 1000  # convert it to mV


def zener_maxcurrent(Vz, powerrating):
    """
    Vz: breakdown voltage of zener diode
    powerrating: power rating of the diode

    Returns the maximum current (in A) passing through the diode
    """
    I_max = powerrating/Vz
    return I_max


def zener_seriesresistor(Vs, Vz, Iz):
    """
    Vs: source voltage
    Vz: breakdown voltage of zener diode
    Iz: max. current allowed to pass through the diode

    Returns the value of the series-resistor to limit
    current to Iz
    """
    Rs = (_normalizevalue(Vs) - _normalizevalue(Vz)) / _normalizevalue(Iz)
    return _Res(Rs)


def capacitor_film_decodevalue(value):
    """
    The last 3 numbers (X) of the code

    140-pF 2A XXX J

    """
    valuestr = str(value)
    expon = int(valuestr[-1])
    pF = int(valuestr[:-1]) * (10**expon)
    if pF > 1000000:
        return "%.2f uF" % (pF / 1000000)
    elif pF > 1000:
        return "%.2f nF" % (pF / 1000)
    else:
        return "%d pF" % pF


def rc_impedance(R, X):
    """
    Calculate the impedance of a circuit using
    resistors and capacitors in series

    R: resistance of a circuit
    X: capacitive reactance of a circuit
       X can also be a tuple (cap, freq) --> see capacitive_reactance

    see https://www.youtube.com/watch?v=xyMH8wKK-Ag
    """
    if isinstance(X, tuple):
        X = capacitive_reactance(X[0], X[1])
    Z = math.sqrt(R**2 + X**2)
    return _Res(Z)
