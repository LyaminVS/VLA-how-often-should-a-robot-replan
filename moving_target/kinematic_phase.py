"""Exploratory fixed-amplitude phase-shift follow-up to the registered sweep."""
import math
from .kinematic import KinematicTargetEnvironment

def phase_reference(elapsed_s,period_s,amplitude_m,start_delay_s,ramp_s,phase_rad):
    tau=elapsed_s-start_delay_s
    if period_s is None or tau<=0:return 0.,0.
    omega=2*math.pi/period_s;u=min(tau/ramp_s,1.);ramp=u*u*(3-2*u);derivative=6*u*(1-u)/ramp_s if u<1 else 0.
    arg=omega*tau+phase_rad
    return amplitude_m*ramp*math.sin(arg),amplitude_m*(derivative*math.sin(arg)+ramp*omega*math.cos(arg))

class PhaseShiftTargetEnvironment(KinematicTargetEnvironment):
    def _reference(self):
        dx,v=phase_reference(self._elapsed(),self.motion_spec.periods_s[self.condition],self.motion_spec.amplitude_m,self.motion_spec.start_delay_s,self.motion_spec.ramp_s,math.pi/2)
        return [self._origin[0]+dx,self._origin[1],self._origin[2]],v
    def telemetry(self):
        data=super().telemetry()
        if data:data['trajectory_phase_rad']=math.pi/2
        return data
