import numpy as np
import os

from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator

vector = {}
def hook_fn(module, inp, out):
    vector['latent_space'] = inp[0].detach().cpu().numpy()

#####
predictor = pretrained_mlip.get_predict_unit('uma-s-1p2', device = 'cuda')
# Backbone
hook_hanlde = predictor.model.module.output_heads.energyandforcehead.head.energy_block.register_forward_hook(hook_fn)
# Last-layer
hook_hanlde = predictor.model.module.output_heads.energyandforcehead.head.energy_block[-1].register_forward_hook(hook_fn)
#####

#####
predictor = pretrained_mlip.load_predict_unit('/your/finetuned/model', device = 'cuda')
# Backbone
hook_hanlde = predictor.model.module.output_heads.efs.energy_block.register_forward_hook(hook_fn)
# Last-layer
hook_hanlde = predictor.model.module.output_heads.efs.energy_block[-1].register_forward_hook(hook_fn)
#####

calc = FAIRChemCalculator(predictor, task_name = 'omat')

desc = []
traj = read(f'../workflow/md/uma-1p2-omat/results/OTAQ/1.traj', index = ':')
for atoms in traj:
    atoms.calc = calc
    _ = atoms.get_potential_energy()
    vec = np.dot(atoms.get_atomic_numbers(), vector['latent_space'])
    desc.append(list(vec))

with open(f'../workflow/md/uma-1p2-omat/descriptors/OTAQ.npy', 'wb') as f:
    np.save(f, desc)
