# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, os, sys, logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import awkward, uproot_methods
from collections import OrderedDict

RANDOM_STATE = 41
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
MODEL_DIR = sys.argv[3]


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def transform(PX, PY, PZ, PENERGY, PMASS, JX, JY, JZ, JENERGY, JMASS, CATEGORIES, EVENTIDS):
    v = OrderedDict()

    mask = PENERGY>0
    n_particles = np.sum(mask, axis=1)

    PX = awkward.JaggedArray.fromcounts(n_particles, PX[mask])
    PY = awkward.JaggedArray.fromcounts(n_particles, PY[mask])
    PZ = awkward.JaggedArray.fromcounts(n_particles, PZ[mask])
    PENERGY = awkward.JaggedArray.fromcounts(n_particles, PENERGY[mask])
    PMASS = awkward.JaggedArray.fromcounts(n_particles, PMASS[mask])
    CATEGORIES = awkward.JaggedArray.fromcounts(n_particles, CATEGORIES[mask])

    JX = awkward.JaggedArray.fromcounts(n_particles, JX[mask])
    JY = awkward.JaggedArray.fromcounts(n_particles, JY[mask])
    JZ = awkward.JaggedArray.fromcounts(n_particles, JZ[mask])
    JENERGY = awkward.JaggedArray.fromcounts(n_particles, JENERGY[mask])
    JMASS = awkward.JaggedArray.fromcounts(n_particles, JMASS[mask])

    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(PX, PY, PZ, PENERGY)
    pt = p4.pt
    jet_p4 = p4.sum()

    v['event_id'] = EVENTIDS

    v['px'] = PX
    v['py'] = PY
    v['pz'] = PZ
    v['penergy'] = PENERGY
    v['pmass'] = PMASS
    v['pcategory'] = CATEGORIES

    v['jx'] = JX
    v['jy'] = JY
    v['jz'] = JZ
    v['jenergy'] = JENERGY
    v['jmass'] = JMASS  

    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi

    v['part_pt_log'] = np.log(pt)
    v['part_ptrel'] = pt/v['jet_pt']
    v['part_logptrel'] = np.log(v['part_ptrel'])

    v['part_e_log'] = np.log(PENERGY)
    v['part_erel'] = PENERGY/jet_p4.energy
    v['part_logerel'] = np.log(v['part_erel'])

    v['part_raw_etarel'] = (p4.eta - v['jet_eta'])
    _jet_etasign = np.sign(v['jet_eta'])
    _jet_etasign[_jet_etasign==0] = 1
    v['part_etarel'] = v['part_raw_etarel'] * _jet_etasign

    v['part_phirel'] = p4.delta_phi(jet_p4)
    v['part_deltaR'] = np.hypot(v['part_etarel'], v['part_phirel'])
    return v

def to_akwd(events):
    max_particles = 200
  
    PX = []
    PY = []
    PZ = []
    PENERGY = []
    PMASS = []

    JX = []
    JY = []
    JZ = []
    JENERGY = []
    JMASS = []

    CATEGORIES = []
    EVENTIDS = []
    feat_cols = ['particle_px',	'particle_py','particle_pz','particle_energy','particle_mass','jet_px','jet_py','jet_pz','jet_energy','jet_mass','particle_category']
    n_features = len(feat_cols)
    for df in events:
        df = df[1]
        arr = np.zeros([max_particles,n_features])
        max_ix = min(len(df),max_particles)
        arr[:max_ix] = df[feat_cols].values[:max_ix]
        PX.append(arr[:,0])
        PY.append(arr[:,1])
        PZ.append(arr[:,2])
        PENERGY.append(arr[:,3])
        PMASS.append(arr[:,4])
        JX.append(arr[:,5])
        JY.append(arr[:,6])
        JZ.append(arr[:,7])
        JENERGY.append(arr[:,8])
        JMASS.append(arr[:,9])
        CATEGORIES.append(arr[:,10])

        EVENTIDS.append(df['event_id'].values[0]) ##single event

    PX = np.asarray(PX)
    PY = np.asarray(PY)
    PZ = np.asarray(PZ)
    PENERGY = np.asarray(PENERGY)
    PMASS = np.asarray(PMASS)
    JX = np.asarray(JX)
    JY = np.asarray(JY)
    JZ = np.asarray(JZ)
    JENERGY = np.asarray(JENERGY)
    JMASS = np.asarray(JMASS)

    CATEGORIES = np.asarray(CATEGORIES)
    EVENTIDS = np.asarray(EVENTIDS)
    v = transform(PX,PY,PZ,PENERGY,PMASS,JX,JY,JZ,JENERGY,JMASS,CATEGORIES,EVENTIDS)
    return v

def to_akwd(events):
    max_particles = 200
  
    PX = []
    PY = []
    PZ = []
    PENERGY = []
    PMASS = []

    JX = []
    JY = []
    JZ = []
    JENERGY = []
    JMASS = []

    CATEGORIES = []
    EVENTIDS = []
    feat_cols = ['particle_px','particle_py','particle_pz','particle_energy','particle_mass','jet_px','jet_py','jet_pz','jet_energy','jet_mass','particle_category']
    n_features = len(feat_cols)
    for df in events:
        df = df[1]
        arr = np.zeros([max_particles,n_features])
        max_ix = min(len(df),max_particles)
        arr[:max_ix] = df[feat_cols].values[:max_ix]
        PX.append(arr[:,0])
        PY.append(arr[:,1])
        PZ.append(arr[:,2])
        PENERGY.append(arr[:,3])
        PMASS.append(arr[:,4])
        JX.append(arr[:,5])
        JY.append(arr[:,6])
        JZ.append(arr[:,7])
        JENERGY.append(arr[:,8])
        JMASS.append(arr[:,9])
        CATEGORIES.append(arr[:,10])

        EVENTIDS.append(df['event_id'].values[0]) ##single event

    PX = np.asarray(PX)
    PY = np.asarray(PY)
    PZ = np.asarray(PZ)
    PENERGY = np.asarray(PENERGY)
    PMASS = np.asarray(PMASS)
    JX = np.asarray(JX)
    JY = np.asarray(JY)
    JZ = np.asarray(JZ)
    JENERGY = np.asarray(JENERGY)
    JMASS = np.asarray(JMASS)

    CATEGORIES = np.asarray(CATEGORIES)
    EVENTIDS = np.asarray(EVENTIDS)
    v = transform(PX,PY,PZ,PENERGY,PMASS,JX,JY,JZ,JENERGY,JMASS,CATEGORIES,EVENTIDS)
    return v


df_test_particle = pd.read_csv(f'{INPUT_DIR}/complex_final_test_R04_particle.csv')
df_test_jet = pd.read_csv(f'{INPUT_DIR}/complex_final_test_R04_jet.csv')

df_test_particle = reduce_mem_usage(df_test_particle)
df_test_jet = reduce_mem_usage(df_test_jet)


df_test_particle_jet = df_test_particle.merge(df_test_jet)
df_test_particle_jet.sort_values(by=['event_id','jet_id'],inplace=True)
df_test_particle_jet.reset_index(inplace=True,drop=True)


df_test_particle_jet_gb = list(df_test_particle_jet.groupby('event_id'))


### Convert test dataset to awkd
v_test = to_akwd(df_test_particle_jet_gb)
awkward.save(f'{OUTPUT_DIR}/test.awkd', v_test, mode='w')



def pad_array(a, maxlen, value=0., dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x


class Dataset(object):

    def __init__(self, filepath, feature_dict = {}, label='event_id', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            
            feature_dict['points'] =  ['px', 'jx']
            feature_dict['features'] = ['part_etarel', 'part_phirel','part_deltaR','part_pt_log', 'px','jx','py', 'pz','penergy','pcategory']
            feature_dict['mask'] = ['part_pt_log']

        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            self._label = a[self.label]
            for k in self.feature_dict:
                cols = self.feature_dict[k]
                if not isinstance(cols, (list, tuple)):
                    cols = [cols]
                arrs = []
                for col in cols:
                    if counts is None:
                        counts = a[col].counts
                    else:
                        assert np.array_equal(counts, a[col].counts)
                    arrs.append(pad_array(a[col], self.pad_len))
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
        logging.info('Finished loading file %s' % self.filepath)


    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        else:
            return self._values[key]
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]


test_dataset = Dataset(f'{OUTPUT_DIR}/test.awkd', data_format='channel_last')


# ##### Generate predictions

model1 = load_model(f'{MODEL_DIR}/model_1_1.h5',custom_objects={'LAMB':tfa.optimizers.lamb.LAMB, 'SWA':tfa.optimizers.SWA})
# print(model1.summary())
preds1 = model1.predict(test_dataset.X)

model2 = load_model(f'{MODEL_DIR}/model_1_2.h5',custom_objects={'LAMB':tfa.optimizers.lamb.LAMB, 'SWA':tfa.optimizers.SWA})
# model2.summary()
preds2  = model2.predict(test_dataset.X)

model3 = load_model(f'{MODEL_DIR}/model_2_1.h5',custom_objects={'LAMB':tfa.optimizers.lamb.LAMB, 'SWA':tfa.optimizers.SWA})
# model3.summary()
preds3 = model3.predict(test_dataset.X)

model4 = load_model(f'{MODEL_DIR}/model_2_2.h5',custom_objects={'LAMB':tfa.optimizers.lamb.LAMB, 'SWA':tfa.optimizers.SWA})
# model4.summary()
preds4 = model4.predict(test_dataset.X)

preds_avg = (preds1+preds2+preds3+preds4)/4
y_pred = np.argmax(preds_avg,axis=1)


#print(np.unique(y_pred), y_pred.shape)

df_events = pd.DataFrame()
df_events['event_id'] = test_dataset.y
# df_events.head()


# MAPPED_LABELS = {1: 0, 4: 1, 5: 2, 21: 3}

df_events['label'] = y_pred
df_events['label'] = df_events['label'].astype(np.int8)
MAPPED_LABELS = {0: 1, 1: 4, 2: 5, 3: 21}
df_events['label'] = df_events.label.map(lambda x: MAPPED_LABELS[x])
#df_events.head()

df_submission = pd.merge(df_events, df_test_jet)[['jet_id','label']]


df_submission.rename(columns={'jet_id':'id'},inplace=True)
#df_submission.head()


# df_test_jet.shape[0], df_submission.shape[0]


# In[31]:


df_submission.to_csv(f'{OUTPUT_DIR}/submission.csv',index=False)
