import openvino.runtime as ov
from bandit import OVBandit
from solvers import UpperConfidenceBound
import cv2
import numpy as np

def main():
    core = ov.Core()
    model = core.read_model(model="vanilla/vanilla.onnx")

    port = model.input(0).shape
    print(port)

    bandit = OVBandit([
        core.compile_model(model=model, device_name="CPU"),
        core.compile_model(model=model, device_name="MYRIAD")
       ], UpperConfidenceBound(2))

    test_image = cv2.imread('kanye.jpg')
    test_image = cv2.resize(test_image, (320, 320))
    inp = np.asarray(test_image).transpose((2,0,1))
    inp = np.expand_dims(inp, axis=0)
    print(inp.shape)

    bandit.predict(inp)
    print("ABC")
    bandit.choose(1)
    bandit.predict(inp)
    print("CDE")

if __name__ == '__main__':
    main()