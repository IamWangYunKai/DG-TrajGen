

class PesudoSensor(object):
    def __init__(self, transform, param):
        self.type_id = 'sensor.camera.rgb'
        self.transform = transform
        self.attributes = dict()
        self.attributes['role_name'] = param['role_name']
        self.attributes['image_size_x'] = str( param['img_length'] )
        self.attributes['image_size_y'] = str( param['img_width'] )
        self.attributes['fov'] = str( param['fov'] )
    
    def get_transform(self):
        return self.transform
