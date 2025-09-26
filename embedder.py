from insightface.app import FaceAnalysis

class InsightFaceEmbedder:
    def __init__(self, model_name="buffalo_l", crop_size=(112, 112)):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.crop_size = crop_size

    def get_embedding(self, img):
        faces = self.app.get(img)
        if len(faces) > 0:
            return faces[0].embedding
        return None
