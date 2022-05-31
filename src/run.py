# %%
from homework import dataloader, MODEL_TYPE, train, test, create_model
import torch
import os

# %%
batch_size = 32
trains, tests = dataloader(batch_size, None)

# %%
learn_rate = 0.001
epoch = 20

# %%
alex = create_model(MODEL_TYPE.ALEX)
train(alex, learn_rate, trains, epoch)

hydra = create_model(MODEL_TYPE.HYDRA)
train(hydra, learn_rate, trains, epoch)

learn_rate = 0.0005
hydra_modii = create_model(MODEL_TYPE.HYDRA)
train(hydra_modii, learn_rate, trains, epoch)

# %%
for m in os.listdir('../model/'):
    paramstr = m.rsplit('.', maxsplit=1)[0]
    param = dict([p.split('-') for p in paramstr.split('_')])
    model = create_model(param['model'])
    model.load_state_dict(torch.load(f'../model/{m}'))
    model.eval()
    print('*'*70, m, '*'*70)
    test(model, tests)


