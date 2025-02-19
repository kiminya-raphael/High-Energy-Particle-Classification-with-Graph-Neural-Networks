import pandas as pd, numpy as np
import sys,os,gc,re,glob, time
from datetime import datetime
from sklearn import preprocessing
from sklearn.metrics import f1_score, log_loss
import awkward
import uproot_methods
from collections import OrderedDict
RANDOM_STATE = 41
np.random.seed(RANDOM_STATE)

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

def transform(PX, PY, PZ, PENERGY, PMASS, JX, JY, JZ, JENERGY, JMASS, CATEGORIES, LABELS):
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

  v['label'] = LABELS

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

def to_akwd(events, ix_start, ix_stop):
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
  LABELS = []
  feat_cols = ['particle_px',	'particle_py','particle_pz','particle_energy','particle_mass','jet_px','jet_py','jet_pz','jet_energy','jet_mass','particle_category']
  n_features = len(feat_cols)
  for df in events[ix_start:ix_stop]:
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

    LABELS.append(df['label_1'].values[0]) ##single label

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
  LABELS = np.asarray(LABELS)
  v = transform(PX,PY,PZ,PENERGY,PMASS,JX,JY,JZ,JENERGY,JMASS,CATEGORIES,LABELS)
  return v

DATA_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)

df_train_particle = pd.read_csv(f'{DATA_DIR}/complex_train_R04_particle.csv')
df_train_jet = pd.read_csv(f'{DATA_DIR}/complex_train_R04_jet.csv')

df_train_particle = reduce_mem_usage(df_train_particle)
df_train_jet = reduce_mem_usage(df_train_jet)
df_train_particle_jet = df_train_particle.merge(df_train_jet)
df_train_particle_jet.sort_values(by=['event_id','jet_id'],inplace=True)
df_train_particle_jet.reset_index(inplace=True,drop=True)
del df_train_jet, df_train_particle

TARGET_CLASSES = sorted(df_train_particle_jet.label.unique())
MAPPED_LABELS = { TARGET_CLASSES[i] : i  for i in range(0, len(TARGET_CLASSES) ) }
df_train_particle_jet['label_1'] = df_train_particle_jet.label.map(lambda x: MAPPED_LABELS[x])

df_train_particle_jet = df_train_particle_jet.sort_values(by='event_id').reset_index(drop=True)
df_train_particle_jet_gb = list(df_train_particle_jet.groupby('event_id'))
print('Preparing datasets')
##1st dataset: first len(train)*0.8 samples for training; last len(train)*0.2 samples for validation
n_train = int(len(df_train_particle_jet_gb)*0.8)
v_train = to_akwd(df_train_particle_jet_gb,0,n_train)
awkward.save(f'{OUT_DIR}/train1.awkd', v_train, mode='w')
del v_train; gc.collect()

v_valid = to_akwd(df_train_particle_jet_gb,n_train,len(df_train_particle_jet_gb)+1)
awkward.save(f'{OUT_DIR}/valid1.awkd', v_valid, mode='w')
del v_valid; gc.collect()
print('Processed dataset 1')

##2nd dataset: first len(train)*0.2 samples for validation; last len(train)*0.8 samples for training
n_valid = int(len(df_train_particle_jet_gb)*0.2)
v_valid = to_akwd(df_train_particle_jet_gb,0,n_valid)
awkward.save(f'{OUT_DIR}/valid2.awkd', v_valid, mode='w')
del v_valid; gc.collect()

v_train = to_akwd(df_train_particle_jet_gb,n_valid,len(df_train_particle_jet_gb)+1)
awkward.save(f'{OUT_DIR}/train2.awkd', v_train, mode='w')
print('Processed dataset 2')
