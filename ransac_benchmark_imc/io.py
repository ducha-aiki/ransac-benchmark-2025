# Copyright 2020 Google LLC, University of Victoria, Czech Technical University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#The most of this code is taked from https://github.com/vcg-uvic/image-matching-benchmark
import numpy as np
import h5py
import os
import cv2

def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    if not os.path.isfile(filename):
        print('Cannot find file {}'.format(filename))
        return None
    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key in keys:
            dict_to_load[key] = f[key][()]
    return dict_to_load

def save_h5(dict_to_save, filename):
    '''Saves dictionary to HDF5 file'''
    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])
    return

def get_h_imgpair(key, dataset, split = 'val'):
    DIR = 'homography'
    if dataset == 'EVD':
        img1_fname = f'{DIR}/{dataset}/{split}/imgs/1/' + key.split('-')[0] + '.png'
        img2_fname = f'{DIR}/{dataset}/{split}/imgs/2/' + key.split('-')[0] + '.png'
    elif dataset == 'HPatchesSeq':
        img1_fname = f'{DIR}/{dataset}/{split}/imgs/{key[:-4]}/1.ppm'
        img2_fname = f'{DIR}/{dataset}/{split}/imgs/{key[:-4]}/{key[-1]}.ppm'
    else:
        raise ValueError ('Unknown dataset, try EVD or HPatchesSeq')
    img1 = cv2.cvtColor(cv2.imread(img1_fname), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_fname), cv2.COLOR_BGR2RGB)
    return img1, img2

def get_h_imgpair2(key, DIR):
    if 'EVD' in DIR:
        img1_fname = f'{DIR}/imgs/1/' + key.split('-')[0] + '.png'
        img2_fname = f'{DIR}/imgs/2/' + key.split('-')[0] + '.png'
    elif 'HPatchesSeq' in DIR:
        img1_fname = f'{DIR}/imgs/{key[:-4]}/1.ppm'
        img2_fname = f'{DIR}/imgs/{key[:-4]}/{key[-1]}.ppm'
    else:
        raise ValueError ('Unknown dataset, try EVD or HPatchesSeq')
    img1 = cv2.cvtColor(cv2.imread(img1_fname), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_fname), cv2.COLOR_BGR2RGB)
    return img1, img2

def get_output_dir(problem: str, split: str, method: str, params: dict):
    problem = problem.lower()
    if problem not in ['e', 'f', 'h', 'pnp']:
        raise ValueError(f'{problem} is unknown problem. Try e, f, h, or pnp')
    param_string = ''
    sorted_keys = sorted([str(x) for x in params.keys()])
    for k in sorted_keys:
        param_string += f'_{k}-{str(params[k])}'
    return os.path.join('results', split, problem, method, param_string)