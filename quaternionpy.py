import numpy as np

# Takes in row vector q_in and returns the quaternion of q_in
# with a positive real component as a row vector
def q_prop(q_in):

    if (q_in[3] >= 0):
        return q_in
    else:
        return -1*np.array(q_in)

# Takes in row vector q_in and returns the quaternion inverse
# of q_in as a row vector
def q_inv(q_in):
    inverseT = np.array([-1, -1, -1, 1])
    return q_in*inverseT

# Takes in a row vector q_in and normalizes it such that its
# magnitude is 1
def q_norm(q_in):
    TOL = 1e-8
    q_in = np.array(q_in)
    eps = q_in.dot(q_in) - 1

    if(np.abs(eps) > TOL):
        tmp_max = np.sqrt(1+eps)
        if(TOL > tmp_max):
            tmp_max = TOL
        eps = 1/tmp_max
    
    else:
        eps = 1 - 0.5*eps
    
    q_out = q_in*eps

    return q_prop(q_out)

# Returns the product of two row vector quaternions
def q_q_mult(q_in1, q_in2):
    q_out = np.zeros(4)

    q_out[0] = q_in1[3]*q_in2[0] - q_in1[2]*q_in2[1] + q_in1[1]*q_in2[2] + q_in1[0]*q_in2[3]
    q_out[1] = q_in1[3]*q_in2[1] + q_in1[2]*q_in2[0] + q_in1[1]*q_in2[3] - q_in1[0]*q_in2[2]
    q_out[2] = q_in1[3]*q_in2[2] + q_in1[2]*q_in2[3] - q_in1[1]*q_in2[0] + q_in1[0]*q_in2[1]
    q_out[3] = q_in1[3]*q_in2[3] - q_in1[2]*q_in2[2] - q_in1[1]*q_in2[1] - q_in1[0]*q_in2[0]

    return q_prop(q_norm(q_out))

# Returns the product qinv*v*q, where v is an input row vector
# of length 3, q is an input row vector of length 4, and qinv is the
# quaternion inverse of q
def qinv_v_q_mult(v_in, q_in):
    v_out = np.zeros(3)

    r0 = -q_in[1] * v_in[2] + q_in[2] * v_in[1] + q_in[3] * v_in[0]
    r1 =  q_in[0] * v_in[2] - q_in[2] * v_in[0] + q_in[3] * v_in[1]
    r2 = -q_in[0] * v_in[1] + q_in[1] * v_in[0] + q_in[3] * v_in[2]
    r3  = q_in[0] * v_in[0] + q_in[1] * v_in[1] + q_in[2] * v_in[2]
        
    v_out[0] =   r0 * q_in[3] + r1 * q_in[2] - r2 * q_in[1] + r3 * q_in[0]
    v_out[1] = - r0 * q_in[2] + r1 * q_in[3] + r2 * q_in[0] + r3 * q_in[1]
    v_out[2] =   r0 * q_in[1] - r1 * q_in[0] + r2 * q_in[3] + r3 * q_in[2]

    return v_out
    
# Returns the product q*v*qinv, where v is an input row vector
# of length 3, q is an input row vector of length 4, and qinv is the
# quaternion inverse of q
def q_v_qinv_mult(v_in, q_in):
    return qinv_v_q_mult(v_in, q_inv(q_in))

# Returns q1inv*q_in2 where q1inv is the quaternion inverse of q_in1
def qinv_q_mult(q_in1,q_in2):
    return q_q_mult(q_inv(q_in1), q_in2)


def q2dcm(q):

    T = np.zeros((3, 3))

    T[0,0] = q[3]**2 + q[0]**2 - q[1]**2 - q[2]**2
    T[0,1] = 2*(q[0]*q[1] + q[3]*q[2])
    T[0,2] = 2*(q[0]*q[2] - q[3]*q[1])
    T[1,0] = 2*(q[0]*q[1] - q[3]*q[2])
    T[1,1] = q[3]**2 - q[0]**2 + q[1]**2 - q[2]**2
    T[1,2] = 2*(q[1]*q[2] + q[3]*q[0])
    T[2,0] = 2*(q[0]*q[2] + q[3]*q[1])
    T[2,1] = 2*(q[1]*q[2] - q[3]*q[0])
    T[2,2] = q[3]**2 - q[0]**2 - q[1]**2 + q[2]**2

    return T


# Takes in 3 by 3 matrix T and returns quaternion q
def dcm2q(T):
    # uses row vector instead of column vector. This can also be reshaped later if needed
    q_tmp = np.zeros(4)

    q_tmp[0] = 1+T[0, 0]-T[1, 1]-T[2, 2]
    q_tmp[1] = 1+T[1, 1]-T[0, 0]-T[2, 2]
    q_tmp[2] = 1+T[2, 2]-T[0, 0]-T[1, 1]
    q_tmp[3] = 1+T[0, 0]+T[1, 1]+T[2, 2]

    ind = np.argmax(q_tmp)
    qmax = q_tmp[ind]

    qr3 = 2*np.sqrt(qmax)
    q_tmp[ind] = 0.25*qr3

    if(ind != 3):
        for i in range(0, 3):
            if i != ind:
                q_tmp[i] = (T[i,ind]+T[ind,i])/qr3
        
        i1 = (ind+0)%3+1
        i2 = (ind+1)%3+1
        
        q_tmp[3] = (T[i1,i2]-T[i2,i1])/qr3
    
    else:
        q_tmp[0] = (T[1,2]-T[2,1])/qr3
        q_tmp[1] = (T[2,0]-T[0,2])/qr3
        q_tmp[2] = (T[0,1]-T[1,0])/qr3

    return q_prop(q_tmp)



# testing:

quat = [1, 2, 3, 4]

quatNorm = q_norm(quat)
print(quatNorm)
quatMatrx = q2dcm(quatNorm)
print(quatMatrx)
quat = dcm2q(quatMatrx)
print(quat)