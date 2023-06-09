import numpy as np
import dt_functions as dt
import copy

def I_UED_f(Bfe, Bpt, bragg_pos, hk_idx, i ):
    '''
    calculates a diffraction intensity for L10 Fept given a certan DW coefficients Bfe and BPt
    for a particular Bragg spot from the electron Atomic form factors
    
     
    Input : 
        Bfe : B factor for Fe ( the thing in the DW exponential that doesn't depend on q)
        Bpt : B factor for Pt ( the thing in the DW exponential that doesn't depend on q)
        
        bragg_pos : position of the Bragg Spot in Inverse Angstrom
        hk_idx : miller index h and k of the particular bragg spot: to see if there is constructive or destructive interference
        
    Returns: 
        Intensity in arb. units 
    '''
    # magnitude of q 
    q_tmp = np.sqrt((bragg_pos[i]**2).sum())
    
    #Debye waller factor
    mfe = Bfe*q_tmp**2
    mpt = Bpt*q_tmp**2
    # load atomic form factors
    f_Fe = dt.electron_form_factor(q_tmp, element = 'Fe')
    f_Pt = dt.electron_form_factor(q_tmp, element = 'Pt')
    
    # retrun intensity depending on the hkl index results
    if np.asarray(hk_idx)[i].sum() % 2 : #if odd
        return (np.exp(-mfe)*f_Fe - np.exp(-mpt)*f_Pt )**2
    else:
        return (np.exp(-mfe)*f_Fe + np.exp(-mpt)*f_Pt )**2
    
def I_UED_f_coefs(Bfe, Bpt, K1, K2, bragg_pos, hk_idx, i ):
    '''
    calculates a diffraction intensity for L10 Fept given a certan DW coefficients Bfe and BPt
    for a particular Bragg spot from the electron Atomic form factors
    
     
    Input : 
        Bfe : B factor for Fe ( the thing in the DW exponential that doesn't depend on q)
        Bpt : B factor for Pt ( the thing in the DW exponential that doesn't depend on q)
        
        bragg_pos : position of the Bragg Spot in Inverse Angstrom
        hk_idx : miller index h and k of the particular bragg spot: to see if there is constructive or destructive interference
        
    Returns: 
        Intensity in arb. units 
    '''
    
    # magnitude of q 
    q_tmp = np.sqrt((bragg_pos[i]**2).sum())
    
    #Debye waller factor
    mfe = Bfe*q_tmp**2
    mpt = Bpt*q_tmp**2
    # load atomic form factors
    f_Fe = dt.electron_form_factor(q_tmp, element = 'Fe')
    f_Pt = dt.electron_form_factor(q_tmp, element = 'Pt')
    
    # retrun intensity depending on the hkl index results
    if np.asarray(hk_idx)[i].sum() % 2 : #if odd
        return (K1*(np.exp(-mfe)*f_Fe - np.exp(-mpt)*f_Pt ))**2
    else:
        return (K2*(np.exp(-mfe)*f_Fe + np.exp(-mpt)*f_Pt ))**2
    
   
    
def I_UED_f_coefs2(Bfe, Bpt, K1, K2, bragg_pos, hk_idx, i ):
    '''
    calculates a diffraction intensity for L10 Fept given a certan DW coefficients Bfe and BPt
    for a particular Bragg spot from the electron Atomic form factors
    
     
    Input : 
        Bfe : B factor for Fe ( the thing in the DW exponential that doesn't depend on q)
        Bpt : B factor for Pt ( the thing in the DW exponential that doesn't depend on q)
        
        bragg_pos : position of the Bragg Spot in Inverse Angstrom
        hk_idx : miller index h and k of the particular bragg spot: to see if there is constructive or destructive interference
        
    Returns: 
        Intensity in arb. units 
    '''
    
    # magnitude of q 
    q_tmp = np.linalg.norm(bragg_pos[i], axis = -1  )
    
    #Debye waller factor
    mfe = Bfe*q_tmp**2
    mpt = Bpt*q_tmp**2
    # load atomic form factors
    f_Fe = dt.electron_form_factor(q_tmp, element = 'Fe')
    f_Pt = dt.electron_form_factor(q_tmp, element = 'Pt')
    
    odd_mask = (np.asarray(hk_idx)[i].sum(axis = -1) % 2 ).astype(bool)
    y = np.zeros_like(q_tmp)

    y[odd_mask] = (K1*(np.exp(-mfe[odd_mask])*f_Fe[odd_mask] - np.exp(-mpt[odd_mask])*f_Pt[odd_mask] ))**2
    y[~odd_mask] = (K2*(np.exp(-mfe[~odd_mask])*f_Fe[~odd_mask] + np.exp(-mpt[~odd_mask])*f_Pt[~odd_mask] ))**2
    
    return y   

import numpy as np
def I_UED_f_coefs_from_fits(q_tmp, q_hkl, Bfe, Bpt, K1, K2):
    '''
    I am a bad person... a good person would document this ... 
    '''
    
    # magnitude of q 
#     q_tmp = np.linalg.norm(bragg_pos[i], axis = -1  )
    
    #Debye waller factor
    mfe = Bfe*q_tmp**2
    mpt = Bpt*q_tmp**2
    # load atomic form factors
    f_Fe = dt.electron_form_factor(q_tmp, element = 'Fe')
    f_Pt = dt.electron_form_factor(q_tmp, element = 'Pt')
    
    odd_mask = (np.asarray(hk_idx)[i].sum(axis = -1) % 2 ).astype(bool)
    y = np.zeros_like(q_tmp)

    y[odd_mask] = (K1*(np.exp(-mfe[odd_mask])*f_Fe[odd_mask] - np.exp(-mpt[odd_mask])*f_Pt[odd_mask] ))**2
    y[~odd_mask] = (K2*(np.exp(-mfe[~odd_mask])*f_Fe[~odd_mask] + np.exp(-mpt[~odd_mask])*f_Pt[~odd_mask] ))**2
    
    return y   




#### These ones are for Debye Waller Analysis as shown in Warrens Book 



def I_UED_f_warren_prop2(Bfe, Bpt,K_even, K_odd,bragg_pos, hk_idx, i ):
    '''
    calculates a diffraction intensity for L10 Fept given a certan DW coefficients Bfe and BPt
    for a particular Bragg spot from the electron Atomic form factors
    
     
    Input : 
        Bfe : B factor for Fe ( the thing in the DW exponential that doesn't depend on q)
        Bpt : B factor for Pt ( the thing in the DW exponential that doesn't depend on q)
        
        bragg_pos : position of the Bragg Spot in Inverse Angstrom
        hk_idx : miller index h and k of the particular bragg spot: to see if there is constructive or destructive interference
    
    Returns: 
        Intensity in arb. units 
    '''
    # magnitude of q 
    q_tmp = np.sqrt((bragg_pos[i]**2).sum())
    
    #Debye waller factor
    mfe = Bfe*q_tmp**2
    mpt = Bpt*q_tmp**2
    # load atomic form factors
    f_Fe = dt.electron_form_factor(q_tmp, element = 'Fe')
    f_Pt = dt.electron_form_factor(q_tmp, element = 'Pt')
    
    # retrun intensity depending on the hkl index results
    if np.asarray(hk_idx)[i].sum() % 2 : #if odd
        return (K_odd*( np.exp(-mfe)*f_Fe - np.exp(-mpt)*f_Pt ))**2
    else:
        return (K_even*(np.exp(-mfe)*f_Fe + np.exp(-mpt)*f_Pt ))**2

    
def I_UED_f_warren_120idx(Bfe, Bpt,K_even,K_odd,
                       bragg_q, allBragg_indices_2):
    '''
    calculates a diffraction intensity for L10 Fept given a certan DW coefficients Bfe and BPt
    for a particular Bragg spot from the electron Atomic form factors
    
     
    Input : 
        Bfe : B factor for Fe ( the thing in the DW exponential that doesn't depend on q)
        Bpt : B factor for Pt ( the thing in the DW exponential that doesn't depend on q)
        
        bragg_q : position of the Bragg Spot in Inverse Angstrom
        bragg_hkl : miller index h and k of the particular bragg spot: to see if there is constructive or destructive interference
    
    Returns: 
        Intensity in arb. units 
    '''

    #Debye waller factor
    mfe = Bfe*bragg_q**2
    mpt = Bpt*bragg_q**2
    # load atomic form factors
    f_Fe = dt.electron_form_factor(bragg_q, element = 'Fe')
    f_Pt = dt.electron_form_factor(bragg_q, element = 'Pt')
    
    # retrun intensity depending on the hkl index results
    y_out = np.ones(len(bragg_q))
    odd_mask = (allBragg_indices_2.sum(axis=-1)  %2).astype(bool)
    
    y_out[odd_mask] =  (K_odd *( np.exp(-mfe[odd_mask] )*f_Fe[odd_mask] - np.exp(-mpt[odd_mask])*f_Pt[odd_mask] ))**2
    y_out[~odd_mask] = (K_even*( np.exp(-mfe[~odd_mask])*f_Fe[~odd_mask] + np.exp(-mpt[~odd_mask])*f_Pt[~odd_mask] ))**2
    
    return y_out

def I_UED_f_warren_prop_norm(Bfe, Bpt,BFE0, BPT0, bragg_pos, hk_idx, i ):
    '''
    calculates a diffraction intensity for L10 Fept given a certan DW coefficients Bfe and BPt
    for a particular Bragg spot from the electron Atomic form factors
    
     
    Input : 
        Bfe : B factor for Fe ( the thing in the DW exponential that doesn't depend on q)
        Bpt : B factor for Pt ( the thing in the DW exponential that doesn't depend on q)
        
        bragg_pos : position of the Bragg Spot in Inverse Angstrom
        hk_idx : miller index h and k of the particular bragg spot: to see if there is constructive or destructive interference
    
    Returns: 
        Intensity in arb. units 
    '''
    # magnitude of q 
    q_tmp = np.linalg.norm(bragg_pos[i], axis =-1)
    
    #Debye waller factor
    mfe = Bfe*q_tmp**2
    mpt = Bpt*q_tmp**2
    
    mfe0 = BFE0*q_tmp**2
    mpt0 = BPT0*q_tmp**2

    # load atomic form factors
    f_Fe = dt.electron_form_factor(q_tmp, element = 'Fe')
    f_Pt = dt.electron_form_factor(q_tmp, element = 'Pt')
    
    # retrun intensity depending on the hkl index results
    if np.asarray(hk_idx)[i].sum() % 2 : #if odd
#         return (np.exp(-mfe)*f_Fe - np.exp(-mpt)*f_Pt)**2 / (np.exp(-mfe0)*f_Fe - np.exp(-mpt0)*f_Pt)**2
        return np.exp( 2*(np.log(np.exp(-mpt)*f_Pt-np.exp(-mfe)*f_Fe) - np.log(np.exp(-mpt0)*f_Pt-np.exp(-mfe0)*f_Fe) ) )
    else:
        return np.exp( 2*(np.log(np.exp(-mfe)*f_Fe + np.exp(-mpt)*f_Pt ) - np.log(np.exp(-mfe0)*f_Fe + np.exp(-mpt0)*f_Pt) ) )
#         return (np.exp(-mfe)*f_Fe + np.exp(-mpt)*f_Pt )**2 / (np.exp(-mfe0)*f_Fe + np.exp(-mpt0)*f_Pt)**2
 
    

    
def I_UED_f_warren_prop_norm2(Bfe, Bpt,BFE0, BPT0, q_tmp, hkl):
    '''
    calculates a diffraction intensity for L10 Fept given a certan DW coefficients Bfe and BPt
    for a particular Bragg spot from the electron Atomic form factors
    
     
    Input : 
        Bfe : B factor for Fe ( the thing in the DW exp.onential that doesn't depend on q)
        Bpt : B factor for Pt ( the thing in the DW exponential that doesn't depend on q)
        
        bragg_pos : position of the Bragg Spot in Inverse Angstrom
        hk_idx : miller index h and k of the particular bragg spot: to see if there is constructive or destructive interference
    
    Returns: 
        Intensity in arb. units 
    '''
    # magnitude of q 
#     q_tmp = np.linalg.norm(bragg_pos[i], axis =-1)
    
    #Debye waller factor
    mfe = Bfe*q_tmp**2
    mpt = Bpt*q_tmp**2
    
    mfe0 = BFE0*q_tmp**2
    mpt0 = BPT0*q_tmp**2
    # load atomic form factors
    f_Fe = dt.electron_form_factor(q_tmp, element = 'Fe')
    f_Pt = dt.electron_form_factor(q_tmp, element = 'Pt')
    
    # retrun intensity depending on the hkl index results
    tmp_idx_odd  = np.where(np.sum(hkl, axis = -1) % 2 )
    tmp_idx_even = np.where(~np.sum(hkl, axis = -1) % 2 )
#     print(f'shape q(inside function) is: {q_tmp.shape}')
    out = np.zeros(len(q_tmp))
    
    try:
        out[tmp_idx_odd] = (np.exp(-mfe[tmp_idx_odd])*f_Fe[tmp_idx_odd] - np.exp(-mpt[tmp_idx_odd])*f_Pt[tmp_idx_odd])**2 / (np.exp(-mfe0[tmp_idx_odd])*f_Fe[tmp_idx_odd] - np.exp(-mpt0[tmp_idx_odd])*f_Pt[tmp_idx_odd])**2
    except:
        print('exeption encountered at odd test')
        if (type(q_tmp)== float) or  (type(q_tmp)== int):
            if (np.sum(hkl, axis = -1) % 2) :
                out = (np.exp(-mfe)*f_Fe - np.exp(-mpt)*f_Pt)**2 / (np.exp(-mfe0)*f_Fe - np.exp(-mpt0)*f_Pt)**2
#     except:
#         print('fuck odd ! ')
    try:
        out[tmp_idx_even] = (np.exp(-mfe[tmp_idx_even])*f_Fe[tmp_idx_even] + np.exp(-mpt[tmp_idx_even])*f_Pt[tmp_idx_even])**2 / (np.exp(-mfe0[tmp_idx_even])*f_Fe[tmp_idx_even] + np.exp(-mpt0[tmp_idx_even])*f_Pt[tmp_idx_even])**2
    except:
        print('exeption encountered at even test')
        if (type(q_tmp)== float) or  (type(q_tmp)== int):
            if (~np.sum(hkl, axis = -1) % 2) :
                out = (np.exp(-mfe)*f_Fe + np.exp(-mpt)*f_Pt)**2 / (np.exp(-mfe0)*f_Fe + np.exp(-mpt0)*f_Pt)**2
#     except:
#         print('fuck odd ! ')
        
#     print(f'out shape is: {out.shape}')
#     print(out.shape)
    return out
#     if np.asarray(hk_idx)[i].sum() % 2 : #if odd
    
#         return (np.exp(-mfe)*f_Fe - np.exp(-mpt)*f_Pt)**2 / (np.exp(-mfe0)*f_Fe - np.exp(-mpt0)*f_Pt)**2
#     else:
#         return (np.exp(-mfe)*f_Fe + np.exp(-mpt)*f_Pt )**2 / (np.exp(-mfe0)*f_Fe + np.exp(-mpt0)*f_Pt)**2
     


    
    
 
    
#### This one is important for all symmetries
   
def select_braggs_Bragg(h,k, allBragg_indices_2):
    '''
    Input h,k and give back the indices of allBragg that correspond to it and the equivalent 
    Bragg
    '''
    # here are the equivalent braggs 
    A = np.where((allBragg_indices_2[:,0] == h)&(allBragg_indices_2[:,1] == k))[0][0]
    B = np.where((allBragg_indices_2[:,0] == -(k))&(allBragg_indices_2[:,1] == h))[0][0]
    C = np.where((allBragg_indices_2[:,0] == -(h))&(allBragg_indices_2[:,1] == -(k)))[0][0]
    D = np.where((allBragg_indices_2[:,0] == k)&(allBragg_indices_2[:,1] == -(h)))[0][0]

    E = np.where((allBragg_indices_2[:,0] == k)&(allBragg_indices_2[:,1] == h))[0][0]
    F = np.where((allBragg_indices_2[:,0] ==  -(h))&(allBragg_indices_2[:,1] == k))[0][0]
    G = np.where((allBragg_indices_2[:,0] == -(k))&(allBragg_indices_2[:,1] == -(h)))[0][0]
    H = np.where((allBragg_indices_2[:,0] == h)&(allBragg_indices_2[:,1] == -(k)))[0][0]

    return A, B, C, D, E, F, G, H




def rotate_point_90deg(df, idx, n_iter = 1):
    '''
    rotates particular point in pandas dataframe by 90 deg n_iter times
    '''
    
    for i in np.arange(n_iter):
        tmp_x = copy.deepcopy(df.iloc[idx]['X_pt4_x_coord'])
        tmp_y = copy.deepcopy(df.iloc[idx]['X_pt4_y_coord'])

        df.loc[idx,'X_pt4_x_coord'] = df.loc[idx,'X_pt1_x_coord']
        df.loc[idx,'X_pt4_y_coord'] = df.loc[idx,'X_pt1_y_coord']

        df.loc[idx,'X_pt1_x_coord'] = df.loc[idx,'X_pt2_x_coord']
        df.loc[idx,'X_pt1_y_coord'] = df.loc[idx,'X_pt2_y_coord']

        df.loc[idx,'X_pt2_x_coord'] = df.loc[idx,'X_pt3_x_coord']
        df.loc[idx,'X_pt2_y_coord'] = df.loc[idx,'X_pt3_y_coord']

        df.loc[idx,'X_pt3_x_coord'] = tmp_x
        df.loc[idx,'X_pt3_y_coord'] = tmp_y

        
def mirror_df(df, idx):
    '''
    mirrors particular point in pandas dataframe 
    '''
    
#     for i in np.arange(n_iter):
    tmp_x = copy.deepcopy(df.iloc[idx]['X_pt4_x_coord'])
    tmp_y = copy.deepcopy(df.iloc[idx]['X_pt4_y_coord'])

    df.loc[idx,'X_pt4_x_coord'] = df.loc[idx,'X_pt3_x_coord']
    df.loc[idx,'X_pt4_y_coord'] = df.loc[idx,'X_pt3_y_coord']    
    
    df.loc[idx,'X_pt3_x_coord'] = tmp_x
    df.loc[idx,'X_pt3_y_coord'] = tmp_y
    
    tmp_x = copy.deepcopy(df.iloc[idx]['X_pt2_x_coord'])
    tmp_y = copy.deepcopy(df.iloc[idx]['X_pt2_y_coord'])    
    
    df.loc[idx,'X_pt2_x_coord'] = df.loc[idx,'X_pt1_x_coord']
    df.loc[idx,'X_pt2_y_coord'] = df.loc[idx,'X_pt1_y_coord']    
    
    df.loc[idx,'X_pt1_x_coord'] = tmp_x
    df.loc[idx,'X_pt1_y_coord'] = tmp_y    


    

def apply_symmetry(coord_df, A,B,C,D,E,F,G,H):
    
    print(f'applying symmetries for points {A, B, C, D, E, F, G, H}')
    
    rotate_point_90deg(coord_df, B, n_iter = 1)
    rotate_point_90deg(coord_df, C, n_iter = 2)
    rotate_point_90deg(coord_df, D, n_iter = 3)


    if len(np.unique([A,B,C,D,E,F,G,H])) == len([A,B,C,D,E,F,G,H]):
        print('all bragg spots different, exploiting mirror symmetry')

        mirror_df(coord_df, E)
        rotate_point_90deg(coord_df, F, n_iter = 1)
        mirror_df(coord_df, F)

        rotate_point_90deg(coord_df, G, n_iter = 2)
        mirror_df(coord_df, G)

        rotate_point_90deg(coord_df, H, n_iter = 3)
        mirror_df(coord_df, H)


