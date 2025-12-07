import numpy as np
from src.metrics.base_metric import BaseMetric
from src.model.dnsmos import DNSMOSPredictor

class DNSMosP835Metric(BaseMetric):
    def __init__(self, name, sr=22050):
        super().__init__(name)
        self.predictor = DNSMOSPredictor(
            'metrics/weights/sig_bak_ovr.onnx',
            'metrics/weights/model_v8.onnx'
        )
        self.sr = sr
    
    def __call__(self, gen_wav, **batch) -> float:
        ovrl_scores = []
        for gen_wav_ in gen_wav:
            d = self.predictor(
                gen_wav_.cpu().numpy().squeeze(),
                self.sr, is_personalized_MOS=False
            )
            ovrl_scores.append(d['OVRL'])
        return float(np.mean(ovrl_scores))
