# -*- coding: utf-8 -*-
"""
Autor:
    Andrés lópez joya
Fecha:
    Noviembre/2019
Contenido:
   
    Inteligencia de Negocio
    Grado en IngenierÃ­a InformÃ¡tica
    Universidad de Granada
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor

def normalizar(valores):
    c1 = (valores - valores.min()) * 1.0
    c2 = (valores.max() - valores.min())
    return c1 / c2

le = preprocessing.LabelEncoder()
#

'''
lectura de datos
'''
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('nepal_earthquake_tra.csv')
data_y = pd.read_csv('nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('nepal_earthquake_tst.csv')

#VISUALIZACIÓN DE LOS DATOS
'''
print("Valores perdidos en x:")

print(data_x.isnull().sum())
'''
'''
print("TIPOS")

print(data_x.dtypes)
'''
'''
print("Valores perdidos en tst:")

print(data_x_tst.isnull().sum())
'''

'''
print("Desequilibrio de valores:")
data_y.damage_grade.value_counts().plot(kind='bar')
plt.xticks(rotation = 0)
plt.show()
'''
print('geo_level_1_id:\n')
print(data_x['geo_level_1_id'].value_counts()[0:6])
print('\ngeo_level_2_id:\n')
print(data_x['geo_level_2_id'].value_counts()[0:6])
print('\ngeo_level_3_id:\n')
print(data_x['geo_level_3_id'].value_counts()[0:6])
print('\ncount_floors_pre_eq:\n')
print(data_x['count_floors_pre_eq'].value_counts()[0:6])
print('\nage:\n')
print(data_x['age'].value_counts()[0:6])
print('\narea_percentage:\n')
print(data_x['area_percentage'].value_counts()[0:6])
print('\nheight_percentage:\n')
print(data_x['height_percentage'].value_counts()[0:6])

print('land_surface_condition:\n')
print(data_x['land_surface_condition'].value_counts()[0:6])
print('\nfoundation_type:\n')
print(data_x['foundation_type'].value_counts()[0:6])
print('\nroof_type:\n')
print(data_x['roof_type'].value_counts()[0:6])
print('\nground_floor_type:\n')
print(data_x['ground_floor_type'].value_counts()[0:6])
print('\nother_floor_type:\n')
print(data_x['other_floor_type'].value_counts()[0:6])
print('\nposition:\n')
print(data_x['position'].value_counts()[0:6])
print('\nplan_configuration:\n')
print(data_x['plan_configuration'].value_counts()[0:6])

print('has_superstructure_adobe_mud:\n')
print(data_x['has_superstructure_adobe_mud'].value_counts()[0:6])
print('\nhas_superstructure_mud_mortar_stone:\n')
print(data_x['has_superstructure_mud_mortar_stone'].value_counts()[0:6])
print('\nhas_superstructure_stone_flag:\n')
print(data_x['has_superstructure_stone_flag'].value_counts()[0:6])
print('\nhas_superstructure_cement_mortar_stone:\n')
print(data_x['has_superstructure_cement_mortar_stone'].value_counts()[0:6])
print('\nhas_superstructure_mud_mortar_brick:\n')
print(data_x['has_superstructure_mud_mortar_brick'].value_counts()[0:6])
print('\nhas_superstructure_cement_mortar_brick:\n')
print(data_x['has_superstructure_cement_mortar_brick'].value_counts()[0:6])
print('\nhas_superstructure_timber:\n')
print(data_x['has_superstructure_timber'].value_counts()[0:6])

print('has_superstructure_bamboo:\n')
print(data_x['has_superstructure_bamboo'].value_counts()[0:6])
print('\nhas_superstructure_rc_non_engineered:\n')
print(data_x['has_superstructure_rc_non_engineered'].value_counts()[0:6])
print('\nhas_superstructure_rc_engineered:\n')
print(data_x['has_superstructure_rc_engineered'].value_counts()[0:6])
print('\nhas_superstructure_other:\n')
print(data_x['has_superstructure_other'].value_counts()[0:6])
print('\nlegal_ownership_status:\n')
print(data_x['legal_ownership_status'].value_counts()[0:6])
print('\ncount_families:\n')
print(data_x['count_families'].value_counts()[0:6])
print('\nhas_secondary_use:\n')
print(data_x['has_secondary_use'].value_counts()[0:6])

print('has_secondary_use_agriculture:\n')
print(data_x['has_secondary_use_agriculture'].value_counts()[0:6])
print('\nhas_secondary_use_hotel:\n')
print(data_x['has_secondary_use_hotel'].value_counts()[0:6])
print('\nhas_secondary_use_rental:\n')
print(data_x['has_secondary_use_rental'].value_counts()[0:6])
print('\nhas_secondary_use_institution:\n')
print(data_x['has_secondary_use_institution'].value_counts()[0:6])
print('\nhas_secondary_use_school:\n')
print(data_x['has_secondary_use_school'].value_counts()[0:6])
print('\nhas_secondary_use_industry:\n')
print(data_x['has_secondary_use_industry'].value_counts()[0:6])
print('\nhas_secondary_use_health_post:\n')
print(data_x['has_secondary_use_health_post'].value_counts()[0:6])

print('has_secondary_use_gov_office:\n')
print(data_x['has_secondary_use_gov_office'].value_counts()[0:6])
print('\nhas_secondary_use_use_police:\n')
print(data_x['has_secondary_use_use_police'].value_counts()[0:6])
print('\nhas_secondary_use_other:\n')
print(data_x['has_secondary_use_other'].value_counts()[0:6])

#PREPROCESADO
#se quitan las columnas que no son muy clasificatorias
print("  Borrando columnas...")
columns_to_drop = ['building_id', 'has_secondary_use_use_police', 'has_secondary_use_gov_office',
                   'has_secondary_use_health_post','has_secondary_use_school','has_secondary_use_institution',
                   'has_secondary_use_industry']
data_x.drop(labels=columns_to_drop, axis=1, inplace = True)

data_x_tst.drop(labels=columns_to_drop, axis=1,inplace = True)
data_y.drop(labels=['building_id'], axis=1,inplace = True)


'''
Se convierten las variables categÃ³ricas a variables numÃ©ricas (ordinales)
'''
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn import preprocessing
mask = data_x.isnull()

data_x_tmp = data_x.fillna(9999)

data_x_tmp = data_x.astype(str).apply(LabelEncoder().fit_transform)

data_x_nan = data_x_tmp.where(~mask, data_x)

mask = data_x_tst.isnull() #mÃ¡scara para luego recuperar los NaN
data_x_tmp = data_x_tst.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado

data_x_tst_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categÃ³ricas en numÃ©ricas

data_x_tst_nan = data_x_tst_tmp.where(~mask, data_x_tst) #se recuperan los NaN



#------------------------------------------------------------------
data_x_norma = data_x_nan.apply(normalizar)
data_x_tst_norma = data_x_tst_nan.apply(normalizar)

'''
data_x_norma,data_x_tst_norma = aplicarnormalizar(data_x_nan,data_x_tst_nan)
'''
X = data_x_norma.values
X_tst = data_x_tst_norma.values
y = np.ravel(data_y.values)


#------------------------------------------------------------------------
'''
ValidaciÃ³n cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
le = preprocessing.LabelEncoder()

from sklearn.metrics import f1_score

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []

    for train, test in cv.split(X, y):
        X_train = X[train]
        y_train, y_test = y[train], y[test]
        
        
        
        t = time.time()
        
        modelo = modelo.fit(X_train,y_train)
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("F1 score (tst): {:.4f}, tiempo: {:6.2f} segundos".format(f1_score(y_test,y_pred,average='micro') , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all
#------------------------------------------------------------------------

'''
print("------ XGB...")
clf = xgb.XGBClassifier(n_estimators = 200)
#clf, y_test_clf = validacion_cruzada(clf,X,y,skf)
#'''

#'''
'''----------------SUBMISSION-1-------------------'''
print("-----RANDOM FOREST-----")
lgbm =RandomForestClassifier(n_estimators=500,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=8,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=True,
                 n_jobs=-1,
                 random_state=1,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None)

lgbm, y_test_lgbm = validacion_cruzada(lgbm,X,y,skf)
clf = lgbm
clf = clf.fit(X,y)
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission2.csv", index=False)

'''----------------SUBMISSION-2-------------------
print("-----KNN-----")
knn = KNeighborsClassifier(n_neighbors=3)

knn, y_test_lgbm = validacion_cruzada(knn,X,y,skf)
clf = knn
clf = clf.fit(X,y)
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission1.csv", index=False)
'''
'''----------------SUBMISSION-3-------------------
print("-----XGB-----")
lgbm =xgb.XGBClassifier(n_estimators = 200)

lgbm, y_test_lgbm = validacion_cruzada(lgbm,X,y,skf)
clf = lgbm
clf = clf.fit(X,y)
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission3.csv", index=False)
'''
'''----------------SUBMISSION-4-------------------
print("-----GRADIENT BOOSTING-----")
lgbm =GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='deprecated', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0)

lgbm, y_test_lgbm = validacion_cruzada(lgbm,X,y,skf)
clf = lgbm
clf = clf.fit(X,y)
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission4.csv", index=False)
'''
'''----------------SUBMISSION-5-------------------
print("-----EXTRA TREE-----")
lgbm =ExtraTreesClassifier(n_estimators=320,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=8,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=True,
                 n_jobs=-1,
                 random_state=1,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None
                 )

lgbm, y_test_lgbm = validacion_cruzada(lgbm,X,y,skf)
clf = lgbm
clf = clf.fit(X,y)
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission5.csv", index=False)
'''
'''
print("-----SVM-----")
lgbm =svm.SVC(decision_function_shape='ovo')

lgbm, y_test_lgbm = validacion_cruzada(lgbm,X,y,skf)
clf = lgbm
clf = clf.fit(X,y)
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission5.csv", index=False)
'''