import numpy as np 
import pandas as pd



form_factor_table = pd.read_csv('/cds/home/d/diegotur/UED/FePt/ext_data/Atomic_form_factor_coeficients.csv', delimiter = ';')
for i in range(form_factor_table['Element'].shape[0]):
    form_factor_table['Element'][i] = form_factor_table['Element'][i].strip()

    
    
def form_factor(Q, element = 'Pt'):
    '''
    retrives the element from the table and gives the atomic form factor for given values of 
    Q in A^-1 
    '''

    element_table = form_factor_table[form_factor_table['Element'] == element]
    
    f = element_table['c'].astype("float").values[0]
    for i in range(1,5):
        ai = element_table[f'a{i}'].astype("float").values[0]
        bi = element_table[f'b{i}'].astype("float").values[0]

        f+= ai*np.exp(-bi*((Q/(4*np.pi))**2) )
    return f

def form_factor2(s, element = 'Pt'):
    '''
    retrives the element from the table and gives the atomic form factor for given values of 
    s in A^-1 
    '''

    element_table = form_factor_table[form_factor_table['Element'] == element]
    
    f = element_table['c'].astype("float").values[0]
    for i in range(1,5):
        ai = element_table[f'a{i}'].astype("float").values[0]
        bi = element_table[f'b{i}'].astype("float").values[0]

        f+= ai*np.exp(-bi*((s)**2) )
    return f

# first I need the periodic table of the elements to get Z.... 
periodic_table = pd.read_csv('/cds/home/d/diegotur/UED/FePt/ext_data/Periodic Table of Elements.csv')

def electron_form_factor(Q, element):
    '''
    first calculates the x-ray atomic form factor and then
    uses the Mott-Bethe fromula to guesstimate the electron scattering form factor for said atom 
    Q in A^-1 
    '''
    constant = 0.23933754 #nm-1
    constant *= 10 #A-1 
    
    f_x = form_factor(Q, element = element)
    Z = periodic_table[periodic_table['Symbol'] ==element]['AtomicNumber'].values[0]
    f_e = constant*( (Z - f_x)/(Q**2) )
    
    return f_e


def electron_form_factor2(s, element):
    '''
    first calculates the x-ray atomic form factor and then
    uses the Mott-Bethe fromula to guesstimate the electron scattering form factor for said atom 
    Q in A^-1 
    '''
    constant = 0.023933754 #nm-1
#     constant *= 10 #A-1 
    
#     s = Q/(4*np.pi)
    
    f_x = form_factor2(s, element = element)
    Z = periodic_table[periodic_table['Symbol'] ==element]['AtomicNumber'].values[0]
    f_e = constant*( (Z - f_x)/(s**2) )
    
    return f_e


def I_UED_BCT(q, h,k,l, Bfe=0, Bpt=0):
    '''
    calculates a diffraction intensity for L10 Fept given a certan DW coefficients Bfe and BPt
    for a particular Bragg spot from the electron Atomic form factors
    
     
    Input : 
        q : 2-3D numpy array with the position of the bragg spot in inverse angstorm
        
        Bfe : B factor for Fe ( the thing in the DW exponential that doesn't depend on q)
        Bpt : B factor for Pt ( the thing in the DW exponential that doesn't depend on q)
         
    Returns: 
        Intensity in arb. units 
    '''
    # magnitude of q 
    q_norm = np.linalg.norm(q)
    
    #Debye waller factor
    mfe = Bfe*q_norm**2
    mpt = Bpt*q_norm**2
    # load atomic form factors
    f_Fe = electron_form_factor(q_norm, element = 'Fe')
    f_Pt = electron_form_factor(q_norm, element = 'Pt')
    
    # retrun intensity depending on the hkl index results
#     print(h,k,l)
    if (h+k+l)%2 !=0 : #if odd
#         print('odd')
        return (np.exp(-mfe)*f_Fe - np.exp(-mpt)*f_Pt )**2
    else:
#         print('even') 
        return (np.exp(-mfe)*f_Fe + np.exp(-mpt)*f_Pt )**2
    




def exp_sat(time, A, B ):
    '''
    single exponential saturation:
    
    Parameters: 
    time: array with time delays
    A: Amplitude 
    B: 1/Tau: time constant
    
    Retruns:
    y : intensities
    '''
    t0= 0 
    y = A-A*np.exp(-B*(time-t0)) 
    
   
    mask = np.heaviside(time-t0, 0 )
    y *= mask
    return y

def exp_sat_tau(time, A, T ):
    '''
    single exponential saturation:
    
    Parameters: 
    time: array with time delays
    A: Amplitude 
    B: 1/Tau: time constant
    
    Retruns:
    y : intensities
    '''
    t0= 0 
#     y = A-A*np.exp(-(time-t0)/T) 
    y = A*(1-np.exp(-(time-t0)/T))
    
    
    
#     mask = (time-t0) < 0
    mask = np.heaviside(time-t0, 0 )
#     y[mask] = 0
    y *= mask
    return y


def two_expssum_free(time, A1, B1, A2, B2):
    '''
    creates a saturation curve that starts at zero
    in the from A1 exp(-t*B1) + A2 exp(-t*B2) 
    
    Parameters:
    time: array with time delays
    A1: Amplitude 1
    B1: 1/Tau 1 
    A2: Amplitude 2
    B2: 1/Tau 2 
    
    Returns: 
    y: array with the intensities
    '''
    exp1 =  exp_sat(time,A1,  B2)
    exp2 = exp_sat(time,A2, B2)
    return exp1 + exp2

def two_expssum_2_with_rec(time,A1, B1, A2, B2,A_rec, B_rec):
    exp1 =  exp_sat(time,A1, B1)
    exp2 =  exp_sat(time,A2, B2)
    exp_rec =  -exp_sat(time,A_rec, B_rec)

    return exp1 + exp2 + exp_rec

def two_expssum_2_with_rec_tau(time,A1, T1, A2, T2,A_rec, T_rec):
    exp1 =  exp_sat_tau(time,A1, T1)
    exp2 =  exp_sat_tau(time,A2, T2)
    exp_rec =  -exp_sat_tau(time,A_rec, T_rec)

    return exp1 + exp2 + exp_rec


def two_expssum_2_with_rec2(time,A1, B1, A2, B2,A_rec, B_rec):
    exp1 =  exp_sat(time,A1, B1)
    exp2 =  exp_sat(time,A2, B2)
    exp_rec =  exp_sat(time,A_rec, B_rec)

    return exp1 + exp2 + exp_rec


######################################################################################################
# calculating lattice parameter and error propagation
######################################################################################################

def find_q(bragg_pos, center_pos, coef=1):
    '''
    takes a bragg postition in px and the center of the diffraction pattern coordinate in px and transforms 
    returns a list of transfer vave vectors
    '''
    q2 = (bragg_pos - center_pos)[0]**2 + (bragg_pos - center_pos)[1]**2
    q = np.sqrt(q2)
    return q*coef


def find_a(bragg_pos, center_pos, bragg_idxs, coef=1):
    '''
    Takes a bragg postition in px and the center of the diffraction pattern coordinate in px and transforms 
    returns the reciprocal lattice parameter by using the index of the bragg spot
    Comment: there is the error function later, this probably should be implemented into a single function with the 
    return_error = False/True parameter at some later point in time
    
    Paramters:
    bragg_pos: coordinate in px array with shape (n,2) n being the number of bragg spots fitted
    center_pos: coordinate in px of the center of the diffraction pattern
    bragg_idxs: index of the each bragg, shape (n,3) with one point being (1, 0, 0) and so on
    
    Returns:
    a: the reciprocal lattice unit vector or a list of recirpocal lattice unit vecotrs
    
    '''
    q2 = (bragg_pos - center_pos)[0]**2 + (bragg_pos - center_pos)[1]**2
    idx2 = bragg_idxs[0]**2 + bragg_idxs[1]**2 + bragg_idxs[2]**2
    a = np.sqrt(q2/idx2)
    return a*coef

def find_q_error(bragg_pos, center_pos, center_error, pos_error, coef=1):
    '''
    Finds the error on the q position from the fitted positions (in px)
    and the fit errors
    Parameters:
    bragg_pos: coordinate in px array with shape (n,2) n being the number of bragg spots fitted
    center_pos: coordinate in px of the center of the diffraction pattern
    center_error: error obtained from the fit on the center coordinate in px
    pos_error : error obtained from the fit on the peak position coordinate in px
    coef: coeficient of proportionality ( we are in small angle approximation) between pixel and wavevector value in A-1
    
    Returns:
    q_error : vector of size n with the estimated errors of the transfer wavevector in A-1
    
    '''
    q = np.sqrt((bragg_pos - center_pos)[0]**2 + (bragg_pos - center_pos)[1]**2)
    
    dq_dx  = (bragg_pos - center_pos)[0] / q
    dq_dcx = (center_pos - bragg_pos)[0] / q
    dq_dy  = (bragg_pos - center_pos)[1] / q
    dq_dcy = (center_pos - bragg_pos)[1] / q
    
    q_error = np.sqrt( dq_dx**2 * pos_error[0]**2 + 
                  dq_dcx**2 * center_error[0]**2  + 
                  dq_dy**2 * pos_error[1]**2 + 
                  dq_dcy**2 * center_error[1]**2  )

    return q_error*coef


def find_a_error(bragg_pos, center_pos, bragg_idxs, center_error, pos_error, coef=1):
    '''
    a being the reciporocal lattice parameter, this calcuates the error from calculating a from the 
    fits.
    
    Parameters:
    bragg_pos: coordinate in px array with shape (n,2) n being the number of bragg spots fitted
    center_pos: coordinate in px of the center of the diffraction pattern
    bragg_idxs: index of the each bragg, shape (n,3) with one point being (1, 0, 0) and so on
    center_error: error obtained from the fit on the center coordinate in px
    pos_error : error obtained from the fit on the peak position coordinate in px
    coef: coeficient of proportionality ( we are in small angle approximation) between pixel and wavevector value in A-1
    
    Returns:
    a_error : vector of size n with the estimated errors of the inverse lattice parameter in A-1
    '''

    q = np.sqrt((bragg_pos - center_pos)[0]**2 + (bragg_pos - center_pos)[1]**2)

    dq_dx  = (bragg_pos - center_pos)[0] / q
    dq_dcx = (center_pos - bragg_pos)[0] / q
    dq_dy  = (bragg_pos - center_pos)[1] / q
    dq_dcy = (center_pos - bragg_pos)[1] / q 
    
    
    q_error = np.sqrt( dq_dx**2 * pos_error[0]**2 + 
                  dq_dcx**2 * center_error[0]**2  + 
                  dq_dy**2 * pos_error[1]**2 + 
                  dq_dcy**2 * center_error[1]**2  )
    
    idx2 = bragg_idxs[0]**2 + bragg_idxs[1]**2 + bragg_idxs[2]**2
    q *= coef
    q_error *= coef
    
    a = q/np.sqrt(idx2)
    a_error = q_error/np.sqrt(idx2)
    
    return a_error


def norm_error(timetrace, timetrace_error):
    '''
    calculates error from normalising to the first value
    '''
    return np.abs(timetrace/timetrace[0])* np.sqrt( (timetrace_error/timetrace)**2 + (timetrace_error[0]/timetrace[0])**2)
    

