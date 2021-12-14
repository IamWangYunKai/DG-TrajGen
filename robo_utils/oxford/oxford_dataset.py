import random
import numpy as np
import numpy.linalg as LA
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from robo_utils.oxford.partial_master import PartialDatasetMaster
from robo_utils.oxford.partial_augment import PartialDatasetAugment

class DIMDataset(Dataset):
    def __init__(self, param, mode, opt=None, data_index=None):
        '''
        Load past costmap and future trajectory
        '''
        self.param = param
        self.opt = opt
        self.input_type = None
        self.data_index = data_index
        if data_index is not None:
            self.dataset_master = PartialDatasetMaster(param, data_index)
        else:
            self.dataset_master = PartialDatasetMaster(param)

        self.partial_datasets = self.dataset_master.partial_datasets

        self.train_key_list = self.dataset_master.train_key_list
        self.eval_key_list = self.dataset_master.eval_key_list
        self.test_key_list = self.dataset_master.test_key_list

        if data_index is not None:
            self._train_key_list = []
            for i in range(len(self.train_key_list)):
                key, index = self.train_key_list[i]
                if key == self.data_index:
                    self._train_key_list.append(self.train_key_list[i])
            self.train_key_list = self._train_key_list

        self.num_trajectory = param.net.num_trajectory
        self.normalize_length = param.net.normalize_length
        self.normalize_speed = param.net.normalize_speed
        
        image_transforms = [
            transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.image_transforms = transforms.Compose(image_transforms)


        image_transformsv2 = [
            transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.image_transformsv2 = transforms.Compose(image_transformsv2)

        self.mode = mode
        self.random_index = None
    

    def set_tmp_index(self, index):
        self.tmp_index = index


    def __getitem__(self, pesudo_index):
        if self.mode == 'train':
            key, index = random.choice(self.train_key_list)
            dataset : PartialDatasetAugment = self.partial_datasets[0]
        elif self.mode == 'eval':
            key, index = random.choice(self.eval_key_list)
            dataset : PartialDatasetAugment = self.partial_datasets[key]
            # key, index = self.eval_key_list[self.tmp_index % len(self.eval_key_list)]
        else:
            key, index = random.choice(self.test_key_list)
            dataset : PartialDatasetAugment = self.partial_datasets[key]


        nav = dataset.get_nav_map(dataset.ref_timestamp_array[index])
        nav = self.image_transformsv2(nav)


        image = dataset.get_image(dataset.ref_timestamp_array[index], crop=True)
        image = self.image_transforms(image)

        image = torch.cat((image, nav), 0)
        
        times, x, y, vx, vy = self.dataset_master.get_trajectory(dataset, index)

        times = times.astype(np.float32) / self.opt.max_t

        v_0 = np.sqrt(vx[0]**2+vy[0]**2) / self.opt.max_speed

        fixed_step = times.shape[0]//(self.opt.points_num+1)

        times = times[::fixed_step][1:-1]
        x = x[::fixed_step][1:-1]
        y = y[::fixed_step][1:-1]
        vx = vx[::fixed_step][1:-1]
        vy = vy[::fixed_step][1:-1]

        x /= self.opt.max_dist
        y /= self.opt.max_dist
        vx /= self.opt.max_speed
        vy /= self.opt.max_speed

        xy = torch.FloatTensor([x, -y]).T
        vxy = torch.FloatTensor([vx, -vy]).T
        v0_array = torch.FloatTensor([v_0]*len(x))
        v_0 = torch.FloatTensor([v_0])

        t = torch.FloatTensor(times)

        return {'img': image, 't': t, 
                #'x':x,'y':-y, 'vx':vx,'vy':-vy,
                'v_0':v_0, 'v0_array':v0_array, 
                'xy':xy, 'vxy':vxy,
                'key': key, 'index': index}

    def __len__(self):
        return 100000000000
    
    def set_mode(self, mode):
        self.mode = mode

class DIVADataset(Dataset):
    def __init__(self, param, mode, opt=None):
        '''
        Load past costmap and future trajectory
        '''
        self.param = param
        self.opt = opt
        self.input_type = None
        self.data_index = [0,1,2,3,4]
        self.dataset_master = PartialDatasetMaster(param)

        self.partial_datasets = self.dataset_master.partial_datasets

        self.train_key_list = self.dataset_master.train_key_list
        self.eval_key_list = self.dataset_master.eval_key_list
        self.test_key_list = self.dataset_master.test_key_list

        self._train_key_list = []
        for i in range(len(self.train_key_list)):
            key, index = self.train_key_list[i]
            if key in self.data_index:
                self._train_key_list.append(self.train_key_list[i])
        self.train_key_list = self._train_key_list

        self.num_trajectory = param.net.num_trajectory
        self.normalize_length = param.net.normalize_length
        self.normalize_speed = param.net.normalize_speed
        
        if mode == 'train':
            image_transforms = [
                transforms.Resize((200, 400), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        else:
            image_transforms = [
                transforms.Resize((200, 400), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        image_transformsv2 = [
            transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        self.image_transforms = transforms.Compose(image_transforms)
        self.image_transformsv2 = transforms.Compose(image_transformsv2)


        self.mode = mode
        self.random_index = None
    

    def set_tmp_index(self, index):
        self.tmp_index = index

    def __getitem__(self, pesudo_index):
        while True:
            if self.mode == 'train':
                key, index = random.choice(self.train_key_list)
                dataset : PartialDatasetAugment = self.partial_datasets[key]####
            elif self.mode == 'eval':
                key, index = random.choice(self.eval_key_list)
                dataset : PartialDatasetAugment = self.partial_datasets[key]
                # key, index = self.eval_key_list[self.tmp_index % len(self.eval_key_list)]
            else:
                key, index = random.choice(self.test_key_list)
                dataset : PartialDatasetAugment = self.partial_datasets[key]

            try:
            # if True:
                nav = dataset.get_nav_map(dataset.ref_timestamp_array[index])
                nav = self.image_transformsv2(nav)

                image = dataset.get_image(dataset.ref_timestamp_array[index], crop=True)
                if image is None: continue
                image = self.image_transforms(image)

                images = torch.cat((image, nav), 0)

            except:
                continue

            times, x, y, vx, vy = self.dataset_master.get_trajectory(dataset, index)

            times = times.astype(np.float32) / self.opt.max_t

            v_0 = np.sqrt(vx[0]**2+vy[0]**2) / self.opt.max_speed

            if times.shape[0] < 47:
                print(times.shape[0])
                continue

            fixed_step = times.shape[0]//self.opt.points_num
            try:
                times = times[::fixed_step]
            except:
                print('Error', times.shape[0], self.opt.points_num, fixed_step)
            x = x[::fixed_step]
            y = y[::fixed_step]
            vx = vx[::fixed_step]
            vy = vy[::fixed_step]
            break

        x /= self.opt.max_dist
        y /= self.opt.max_dist
        vx /= self.opt.max_speed
        vy /= self.opt.max_speed

        xy = torch.FloatTensor([x, -y]).T
        vxy = torch.FloatTensor([vx, -vy]).T
        v0_array = torch.FloatTensor([v_0]*len(x))
        v_0 = torch.FloatTensor([v_0])

        domian = [0]*len(self.data_index)
        if self.mode == 'train': domian[key%len(self.data_index)] = 1.
        domian = torch.FloatTensor(domian)

        t = torch.FloatTensor(times)

        return {'img': images, 't': t, 'domian':domian,
                'v_0':v_0, 'v0_array':v0_array, 
                'xy':xy, 'vxy':vxy,
                'key': key, 'index': index}

    def __len__(self):
        return 100000000000
    
    def set_mode(self, mode):
        self.mode = mode

class GANDataset(Dataset):
    def __init__(self, param, mode, opt=None):
        '''
        Load past costmap and future trajectory
        '''
        self.param = param
        self.opt = opt
        self.input_type = None
        self.data_index = [0,1,2,3,4]
        self.dataset_master = PartialDatasetMaster(param)

        self.partial_datasets = self.dataset_master.partial_datasets

        self.train_key_list = self.dataset_master.train_key_list
        self.eval_key_list = self.dataset_master.eval_key_list
        self.test_key_list = self.dataset_master.test_key_list

        self._train_key_list = []
        for i in range(len(self.train_key_list)):
            key, index = self.train_key_list[i]
            if key in self.data_index:
                self._train_key_list.append(self.train_key_list[i])
        self.train_key_list = self._train_key_list

        self.num_trajectory = param.net.num_trajectory
        self.normalize_length = param.net.normalize_length
        self.normalize_speed = param.net.normalize_speed
        
        self.mode = mode
        self.random_index = None
    
    def PJcurvature(self, x,y):
        """
        input  : the coordinate of the three point
        output : the curvature and norm direction
        refer to https://github.com/Pjer-zhang/PJCurvature for detail
        """
        t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
        t_b = LA.norm([x[2]-x[1],y[2]-y[1]])
        
        M = np.array([
            [1, -t_a, t_a**2],
            [1, 0,    0     ],
            [1,  t_b, t_b**2]
        ])

        a = np.matmul(LA.inv(M),x)
        b = np.matmul(LA.inv(M),y)

        kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
        return kappa#, [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)

    def set_tmp_index(self, index):
        self.tmp_index = index

    def __getitem__(self, pesudo_index):
        while True:
            if self.mode == 'train':
                key, index = random.choice(self.train_key_list)
                dataset : PartialDatasetAugment = self.partial_datasets[key]####
            elif self.mode == 'eval':
                key, index = random.choice(self.eval_key_list)
                dataset : PartialDatasetAugment = self.partial_datasets[key]
            else:
                key, index = random.choice(self.test_key_list)
                dataset : PartialDatasetAugment = self.partial_datasets[key]

            times, x, y, vx, vy = self.dataset_master.get_trajectory(dataset, index)

            times = times.astype(np.float32) / self.opt.max_t

            v_0 = np.sqrt(vx[0]**2+vy[0]**2) / self.opt.max_speed

            if v_0 > 4:
                continue

            if times.shape[0] < 47:
                print(times.shape[0])
                continue
            # print(times.shape[0], self.opt.points_num)
            fixed_step = times.shape[0]//self.opt.points_num
            try:
                times = times[::fixed_step]
            except:
                print('Error', times.shape[0], self.opt.points_num, fixed_step)
            x = x[::fixed_step]
            y = y[::fixed_step]
            vx = vx[::fixed_step]
            vy = vy[::fixed_step]

            k = self.PJcurvature([0, x[(2*len(x))//3], x[-1]], [0, y[(2*len(y))//3], y[-1]])
            if abs(k) > 1: k=0
            if random.random() < 0.5:
                if abs(k) < 0.7:
                    continue
            break

        x /= self.opt.max_dist
        y /= self.opt.max_dist
        vx /= self.opt.max_speed
        vy /= self.opt.max_speed



        if random.random() < 0.5:
            # mirror
            xy = torch.FloatTensor([x, y]).T
            vxy = torch.FloatTensor([vx, vy]).T
            k = -k
        else:
            # normal
            xy = torch.FloatTensor([x, -y]).T
            vxy = torch.FloatTensor([vx, -vy]).T


        v0_array = torch.FloatTensor([v_0]*len(x))
        v_0 = torch.FloatTensor([v_0])

        k = torch.FloatTensor(np.array(k).astype(np.float32))
        t = torch.FloatTensor(times)

        return {
                'k':k,
                't': t,
                'v_0':v_0, 'v0_array':v0_array, 
                'xy':xy, 'vxy':vxy,
                'key': key, 'index': index}

    def __len__(self):
        return 100000000000
    
    def set_mode(self, mode):
        self.mode = mode