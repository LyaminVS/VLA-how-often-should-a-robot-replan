import math
import pytest
from moving_target.kinematic_phase import phase_reference
from moving_target.environment import reference_motion

@pytest.mark.parametrize('t',[0.,.5,.51,1.,1.5,2.5,4.])
def test_zero_phase_matches_original(t):
 assert phase_reference(t,4.,.04,.5,1.,0.)==pytest.approx(reference_motion(t,4.,.04,.5,1.))

def test_shifted_phase_is_at_negative_peak_near_original_return_time():
 x,v=phase_reference(2.5,4.,.04,.5,1.,math.pi/2);assert x==pytest.approx(-.04);assert v==pytest.approx(0.,abs=1e-12)

@pytest.mark.parametrize('t',[.51,.8,1.4,1.6,2.4])
def test_velocity_is_derivative_of_position(t):
 h=1e-6;v=phase_reference(t,4.,.04,.5,1.,math.pi/2)[1];finite=(phase_reference(t+h,4.,.04,.5,1.,math.pi/2)[0]-phase_reference(t-h,4.,.04,.5,1.,math.pi/2)[0])/(2*h);assert v==pytest.approx(finite,abs=1e-9)
