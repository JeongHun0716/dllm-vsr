"""Stub for ShuffleNetV2 — only used when backbone_type=='shufflenet'.

dlm-vsr 에서 USR2 huge / large / baseplus / base 는 모두 ResNet 백본을 사용하므로
이 모듈은 import-time 심볼만 충족시키면 된다. 실제로 호출되는 경우는 없다.
필요 시 upstream USR2 의 shufflenetv2.py 를 가져와 채우면 그대로 사용 가능.
"""


class ShuffleNetV2:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ShuffleNetV2 stub: this dlm-vsr build does not include the ShuffleNet "
            "frontend. Use backbone_type='resnet' (default for huge/large/base*)."
        )
