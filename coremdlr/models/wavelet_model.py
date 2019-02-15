"""
TODO: wavelet/shearlet/etc filters for feature extractions.

Note: will have to use Discrete WT for 2D (image) data. CWT can be used for 1D (pGR).
"""
import pywt
import numpy as np
from coremdlr.models import FeatureModel

class WaveletModel(FeatureModel):
    """
    """
    def __init__(self, dataset_cls, dataset_args={}, model_args={}):

        FeatureModel.__init__(self, dataset_cls, dataset_args, model_args)
        if self.was_loaded_from_file:
            return

        self.wavelet_name = self.model_args.get('wavelet', 'morl')
        assert self.wavelet in pywt.wavelist(), 'Invalid `wavelet` name.'
        self.wavelet = pywt.ContinuousWavelet(self.wavelet_name)
        print('Making WaveletModel with wavelet: ', wavelet, end='\n\n')

        min_scale = self.model_args.get('min_scale', 5)
        max_scale = self.model_args.get('max_scale', 5000)
        num_scales = self.model_args.get('num_scales', 25)
        scale_type = self.model_args.get('scale_type', 'lin')
        if 'log' in scale_type.lower():
            self.scales = np.logspace(np.log(min_scale), np.log(max_scale), num_scales)
        else:
            self.scales = np.linspace(min_scale, max_scale, num_scales)
