from .dc_base import DC_BASE
from .text_dc_base import TEXT_DC_BASE
from .audio_dc_base import AUDIO_DC_BASE
from .cafe import CAFE
from .idm import IDM
from .idc import IDC
from .random import Random
from .dual_condensation import DualCondensation
from .dream import DREAM
from .mtt import MTT
from .dream_text import DREAM_Text
from .dual_condensation_text import DualCondensation_Text
from .dream_audio import DREAM_Audio
from .dual_condensation_audio import DualCondensation_Audio

def get_method(name):
    try:
        methods = {
            'Test' : DC_BASE, # Just for evaluation, using base class.
            'Continual' : DC_BASE, # Just for evaluation, using base class.
            'Visualize' : DC_BASE, # Just for evaluation, using base class.
            'Upperbound' : DC_BASE, # Just for evaluation, using base class.
            'Test_Text' : TEXT_DC_BASE, # Just for evaluation, using base class.
            'Continual_Text' : TEXT_DC_BASE, # Just for evaluation, using base class.
            'Visualize_Text' : TEXT_DC_BASE, # Just for evaluation, using base class.
            'Upperbound_Text' : TEXT_DC_BASE, # Just for evaluation, using base class.
            'Test_Audio' : AUDIO_DC_BASE, # Just for evaluation, using base class.
            'Continual_Audio' : AUDIO_DC_BASE, # Just for evaluation, using base class.
            'Visualize_Audio' : AUDIO_DC_BASE, # Just for evaluation, using base class.
            'Upperbound_Audio' : AUDIO_DC_BASE, # Just for evaluation, using base class.
            'DualCondensation' : DualCondensation,
            'DualCondensation_Computation' : DualCondensation,
            'CAFE' : CAFE,
            'CAFE_Computation' : CAFE,
            'IDM' : IDM,
            'IDM_Computation' : IDM,
            'IDC' : IDC,
            'IDC_Computation' : IDC,
            'Random' : Random,
            'Random_Computation' : Random,
            'DREAM' : DREAM,
            'DREAM_Computation' : DREAM,
            'MTT' : MTT,
            'MTT_Computation' : MTT,
            'DREAM_Text' : DREAM_Text,
            'DualCondensation_Text' : DualCondensation_Text,
            'DREAM_Audio' : DREAM_Audio,
            'DualCondensation_Audio' : DualCondensation_Audio,
            # 'DC' : DC,
        }   
        return methods[name]
    except:
        raise NotImplementedError(f"Method {name} not implemented.")