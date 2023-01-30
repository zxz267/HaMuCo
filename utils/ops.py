import numpy as np
import torch


def batch_compute_similarity_transform_numpy(S1, S2, R_GT=None):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2, axis=1).sum(axis=1)

    # 3. The outer product of X1 and X2.
    K = np.matmul(X1, X2.transpose(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.transpose(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[1])[None]
    Z = Z.repeat(U.shape[0], axis=0)
    Z[:, -1, -1] *= np.sign(np.linalg.det(np.matmul(U, V.transpose(0, 2, 1))))

    # Construct R.
    if R_GT is None:
        R = np.matmul(V, np.matmul(Z, U.transpose(0, 2, 1)))
    else:
        R = R_GT

    # 5. Recover scale.
    scale = np.concatenate([np.trace(x)[None] for x in np.matmul(R, K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale[:, None, None] * (np.matmul(R, mu1)))

    # 7. Error:
    S1_hat = scale[:, None, None] * np.matmul(R, S1) + t

    return S1_hat, (scale, R, t)

def batch_compute_similarity_transform_torch(S1, S2, R_GT = None):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # Construct R.
    if R_GT is None:
        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))
        R = V.bmm(Z.bmm(U.permute(0,2,1)))
    else:
        R = R_GT

    # # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
    
    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    return S1_hat, (scale, R, t)

def batch_remove_align(coords, scale, R, t):
    R = R.permute(0, 2, 1)
    coords = coords.permute(0, 2, 1)
    res = (torch.matmul(R, coords) - t) / scale[:, None, None]
    return res.permute(0, 2, 1)

