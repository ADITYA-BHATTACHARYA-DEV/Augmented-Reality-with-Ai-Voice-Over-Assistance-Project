from direct.showbase.ShowBase import ShowBase
# --- Corrected Imports ---
from panda3d.core import (FrameBufferProperties, WindowProperties,
                          GraphicsPipe, Texture, AmbientLight,
                          DirectionalLight, GraphicsOutput)  # Added GraphicsOutput


# --- Inside your __init__ or a helper method ---
def setup_buffer(self, width, height):
    wp = WindowProperties.size(width, height)
    fbp = FrameBufferProperties()
    fbp.setRgba(True)
    fbp.setDepthBits(24)

    # Use makeTextureBuffer for a cleaner approach
    self.buf = self.win.makeTextureBuffer("offscreen", width, height,
                                          tex=None, to_ram=True)

    # Alternatively, if you must use makeOutput:
    # self.buf = self.graphicsEngine.makeOutput(
    #     self.pipe, "offscreen", -2, fbp, wp,
    #     GraphicsPipe.BFRefuseWindow, self.win.getGsg()
    # )
    # self.tex = Texture()
    # self.buf.addRenderTexture(self.tex, GraphicsOutput.RTMCopyRam)
import numpy as np

class CarRenderer(ShowBase):
    def __init__(self, width=640, height=480):
        ShowBase.__init__(self)
        self.disableMouse()

        # Offscreen buffer
        fbp = FrameBufferProperties()
        fbp.setRgba(True); fbp.setDepthBits(24)
        wp  = WindowProperties.size(width, height)
        self.buf = self.graphicsEngine.makeOutput(
            self.pipe, "offscreen", -2, fbp, wp,
            GraphicsPipe.BFRefuseWindow, self.win.getGsg()
        )
        self.tex = Texture()
        self.buf.addRenderTexture(self.tex, GraphicsOutput.RTMCopyRam)
        self.cam2 = self.makeCamera(self.buf)
        self.cam2.setPos(0, -8, 2)
        self.cam2.lookAt(0, 0, 0)

        # Load car model  (place car.glb in project folder)
        self.car = self.loader.loadModel("car.obj")
        self.car.reparentTo(self.render)
        self.car.setScale(1.0)

        # Lighting
        al = AmbientLight("al"); al.setColor((0.4,0.4,0.4,1))
        dl = DirectionalLight("dl"); dl.setColor((0.9,0.9,0.9,1))
        self.render.setLight(self.render.attachNewNode(al))
        dln = self.render.attachNewNode(dl)
        dln.setHpr(45, -60, 0)
        self.render.setLight(dln)

        self.scale      = 1.0
        self.yaw        = 0.0
        self.interior   = False

    def update(self, gesture, scale_delta, rotate_delta):
        self.scale = max(0.3, min(3.0, self.scale + scale_delta))
        self.yaw  += rotate_delta
        self.car.setScale(self.scale)
        self.car.setH(self.yaw)

        if gesture == "interior_view":
            self.cam2.setPos(0, -0.5, 0.5)   # inside cabin
            self.cam2.lookAt(0.5, 0.5, 0.5)
        elif gesture == "reset":
            self.cam2.setPos(0, -8, 2)
            self.cam2.lookAt(0, 0, 0)
            self.scale = 1.0
            self.car.setScale(1.0)

        self.graphicsEngine.renderFrame()

    def get_frame_rgba(self):
        self.tex.store(None)
        data = self.tex.getRamImageAs("RGBA")
        arr  = np.frombuffer(data, dtype=np.uint8)
        h, w = self.tex.getYSize(), self.tex.getXSize()
        return arr.reshape((h, w, 4))[::-1]   # flip vertically