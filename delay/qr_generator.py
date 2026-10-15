import qrcode
import numpy as np
import cv2
import time


from common.pipeline_stage import PipelineStage


class QRGenerator(PipelineStage):
    def __init__(
        self,
        configs,
        logger,
        in_mission,
        in_q=None,
        out_q=None,
        to_update=None,
        idx=0,
    ):
        super().__init__(configs, logger, in_mission, in_q, out_q, to_update, idx)
        self.configs = configs
        self.logger = logger
        self.idx = idx
        self.to_update = to_update
        self.in_mission = in_mission
        self.FPS = 10
        self.FRAME_TIME = 1.0 / self.FPS
        self.qr = qrcode.QRCode(
            version=3,
            error_correction=qrcode.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        self.w = 2000
        self.h = 1000

    def mission(self) -> bool:
        while True:
            while self.in_mission.is_set(self.idx):
                try:
                    frame = self.generate_qr_frame()
                    cv2.namedWindow("QR Generator", cv2.WINDOW_NORMAL)
                    scale = min(self.w, self.h)
                    frame = cv2.resize(
                        frame, (scale, scale), interpolation=cv2.INTER_NEAREST
                    )
                    cv2.imshow("QR Generator", frame)
                    if cv2.waitKey(1) == 27:
                        break
                except Exception as e:
                    self.logger.error(f"{e}")

    def update(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get(self, data) -> None:
        pass

    def put(self, data) -> None:
        pass

    def _dump_data(self) -> None:
        pass

    def generate_qr_frame(self):
        start_time = time.time()

        ts = time.time_ns() // 1000

        # Generate QR code
        self.qr.clear()
        self.qr.add_data(str(ts))
        self.qr.make(fit=True)
        pil_img = self.qr.make_image(fill_color="black", back_color="white")

        # Convert PIL → NumPy array
        img_qr = np.array(pil_img)
        if img_qr.dtype == bool:
            img_qr = img_qr.astype(np.uint8) * 255

        # Convert grayscale → RGB
        if len(img_qr.shape) == 2:
            img_qr = cv2.cvtColor(img_qr, cv2.COLOR_GRAY2BGR)

        # Sleep to maintain FPS
        elapsed = time.time() - start_time
        time.sleep(max(0, self.FRAME_TIME - elapsed))

        return img_qr
