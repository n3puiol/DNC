import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from sklearn.metrics import silhouette_score

from ..builder import HEADS
from .cls_head import ClsHead
import torch.distributed as dist
from mmcv.cnn import ConvModule
import numpy as np
from einops import repeat
from timm.models.layers import trunc_normal_


@torch.no_grad()
def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):  # skinhorn iteration time=3
    # epsilon: convergence parameter
    Q = torch.exp(out / epsilon).t()  # K x B
    B = Q.shape[1]
    K = Q.shape[0]  # how many sub-centriods

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per sub-centriods must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    Q = Q.t()

    indexs = torch.argmax(Q, dim=1)
    Q = torch.nn.functional.one_hot(indexs, num_classes=Q.shape[1]).float()
    return Q, indexs


# handle exceptions
@torch.no_grad()
def approx_ot(M, max_iter=10):
    M = M.t()  # [#Guass, K]
    p = M.shape[0]  # #Guass
    n = M.shape[1]  # K

    r = torch.ones((p,), dtype=torch.float64).to(M.device) / p  # .to(L.device) / K
    c = torch.ones((n,), dtype=torch.float64).to(M.device) / n  # .to(L.device) / B 先不要 等会加上

    B = sinkhorn_knopp(r, c, M, reg=0.05, max_iter=max_iter)

    B = B.t()

    indexes = torch.argmax(B, dim=1)
    G = gumbel_softmax(B, tau=0.5, hard=True)
    if torch.isnan(G).int().sum() > 0:
        print('output has nan, use self_gambel_softmax')

    return G.to(torch.float32), indexes


@torch.no_grad()
def sinkhorn_knopp(a, b, M, reg=0.05, max_iter=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(a) == 0:
        print('ERROR-- should have a value as input')
        a = torch.ones((M.shape[0],), dtype=torch.float64) / M.shape[0]
    if len(b) == 0:
        print('ERROR-- should have b value as input')
        b = torch.ones((M.shape[1],), dtype=torch.float64) / M.shape[1]

    # init data
    dim_a = len(a)
    dim_b = len(b)

    u = torch.zeros(dim_a,
                    dtype=torch.float64).to(device)  # / dim_a; #u = np.log(u) # log to match linear-scale sinkhorn initializations
    v = torch.zeros(dim_b, dtype=torch.float64).to(device)  # / dim_b; #v = np.log(v)

    K = torch.exp(-M / reg).to(device)
    B = torch.einsum('i,ij,j->ij', torch.exp(u), K, torch.exp(v)).to(device)
    ones_a = torch.ones((dim_a,), dtype=torch.float64).to(device)
    ones_b = torch.ones((dim_b,), dtype=torch.float64).to(device)

    cpt = 0

    for _ in range(max_iter):
        u_prev = u
        v_prev = v

        if cpt % 2 == 0:
            u = u_prev + torch.log(a) - torch.log(torch.matmul(ones_b, B.T) + 1e-6)  # transpose-row
            v = v_prev
        else:
            v = v_prev + torch.log(b) - torch.log(torch.matmul(ones_a, B) + 1e-6)
            u = u_prev
        B = torch.einsum('i,ij,j->ij', torch.exp(u), K, torch.exp(v))

        if (torch.any(torch.isnan(u)) or torch.any(torch.isinf(u))
                or torch.any(torch.isnan(v)) or torch.any(torch.isinf(v))
                or torch.any(torch.isnan(B)) or torch.any(torch.isinf(B))):
            # we have reached the machine precision #np.any(KtransposeU == 0)
            # come back to previous solution and quit loop
            u = u_prev
            v = v_prev
            B = torch.einsum('i,ij,j->ij', torch.exp(u), K, torch.exp(v))
            break
        cpt = cpt + 1
    return B


def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """

    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


# K * #Guass as input
@torch.no_grad()
def agd_torch_no_grad_gpu(M, max_iter=20, eps=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    M = M.t()  # [#Guass, K]
    p = M.shape[0]  # #Guass
    n = M.shape[1]  # K

    X = torch.zeros((p, n), dtype=torch.float64).to(device)

    r = torch.ones((p,), dtype=torch.float64).to(M.device) / p  # .to(L.device) / K
    c = torch.ones((n,), dtype=torch.float64).to(M.device) / n  # .to(L.device) / B 先不要 等会加上

    gamma = eps / (3 * math.log(n))

    A = torch.zeros((max_iter, 1), dtype=torch.float64).to(M.device)  # init array of A_k
    L = torch.zeros((max_iter, 1), dtype=torch.float64).to(M.device)  # init array of L_k

    # set initial values for APDAGD
    L[0, 0] = 1  # set L_0

    # set starting point for APDAGD
    y = torch.zeros((n + p, max_iter),
                    dtype=torch.float64).to(device)  # init array of points y_k for which usually the convergence rate is proved (eta)
    z = torch.zeros((n + p, max_iter),
                    dtype=torch.float64).to(device)  # init array of points z_k. this is the Mirror Descent sequence. (zeta)
    j = 0
    # main cycle of APDAGD
    for k in range(0, (max_iter - 1)):
        L_t = (2 ** (j - 1)) * L[k, 0]  # current trial for L
        a_t = (1 + torch.sqrt(1 + 4 * L_t * A[k, 0])) / (
                2 * L_t)  # trial for calculate a_k as solution of quadratic equation explicitly
        A_t = A[k, 0] + a_t  # trial of A_k
        tau = a_t / A_t  # trial of \tau_{k}
        x_t = tau * z[:, k] + (1 - tau) * y[:, k]  # trial for x_k

        lamb = x_t[:n, ]
        mu = x_t[n:n + p, ]

        # 1) [K,1] * [1, #Gauss] --> [K, #Gauss].T -->[#Gauss, K]; 2) [K, 1] * [#Guass, 1].T --> [K, #Guass]--.T--> [#Guass, K]
        M_new = -M - torch.matmul(lamb.reshape(-1, 1).to(device),
                                  torch.ones((1, p), dtype=torch.float64).to(device)).T - torch.matmul(
            torch.ones((n, 1), dtype=torch.float64).to(device), mu.reshape(-1, 1).T.to(device)).T

        X_lamb = torch.exp(M_new / gamma)
        sum_X = torch.sum(X_lamb)
        X_lamb = X_lamb / sum_X
        grad_psi_x_t = torch.zeros((n + p,), dtype=torch.float64).to(device)
        grad_psi_x_t[:p, ] = r - torch.sum(X_lamb, axis=1)
        grad_psi_x_t[p:p + n, ] = c - torch.sum(X_lamb, axis=0).T

        X = tau * X_lamb + (1 - tau) * X  # set primal variable

        L[k + 1, 0] = L_t
        j += 1

    X = X.t()

    indexs = torch.argmax(X, dim=1)
    G = F.gumbel_softmax(X, tau=0.5, hard=True)

    return G.to(torch.float32), indexs  # change into G as well


@HEADS.register_module()
class SubCentroids_Head_Formal(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(SubCentroids_Head_Formal, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.temperature = 0.1
        self.update_subcentroids = True
        self.gamma = 0.999
        self.pretrain_subcentroids = False
        self.use_subcentroids = True
        self.centroid_contrast_loss = False
        self.centroid_contrast_loss_weights = 0.005
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.is_only_cross_entropy = True
        self.memory_bank = []
        self.memory_bank_label = []

        # 500 for swin-T # ResNet for 1000 # for swin-B 300 # 400 for swin-s #1000 for mobilenet-v2
        self.batch_size_num_limit = 1000
        print('batch size limit', self.batch_size_num_limit)

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        convs = [ConvModule(
            512,
            512,  # 2048
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)]
        self.convs = nn.Sequential(*convs)

        # 512 for resnet18, 2048 for resnet50
        # if turn 512, then no dimension expand (2048 for expansion for resnet18)
        # 2048 for imagenet without expanding dimension
        # 1024 for swin transformer without expanding dimension
        embedding_dim = self.in_channels

        # Initialize with maximum possible subcentroids
        self.max_subcentroids = 10
        self.candidate_subcentroids = [2, 4, 6, 8, 10]
        # self.optimal_subcentroids = {k: 4 for k in range(num_classes)}  # Default to 4
        self.optimal_subcentroids = nn.Parameter(torch.zeros(self.num_classes),
                                                 requires_grad=False)
        self.optimal_subcentroids.data.fill_(4)  # Default to 4
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.max_subcentroids, embedding_dim),
                                       requires_grad=self.pretrain_subcentroids)

        # Track best silhouette scores
        # self.best_silhouette = {k: -1.0 for k in range(num_classes)}
        self.best_silhouette = nn.Parameter(torch.zeros(self.num_classes),
                                            requires_grad=False)
        self.best_silhouette.data.fill_(-1.0)  # Initialize with -1

        # CK times embedding_dim
        self.feat_norm = nn.LayerNorm(embedding_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        self.apply(init_weights)
        trunc_normal_(self.prototypes, std=0.02)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.forward(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    @staticmethod
    def momentum_update(old_value, new_value, momentum, debug=False):
        update = momentum * old_value + (1 - momentum) * new_value
        if debug:
            print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
                torch.norm(update, p=2)))
        return update

    @staticmethod
    def compute_silhouette(features, cluster_labels):
        """Compute silhouette score for clustering evaluation.

        Args:
            features (torch.Tensor): Feature embeddings (N x D)
            cluster_labels (torch.Tensor): Cluster assignments (N,)
        Returns:
            float: Silhouette score between -1 and 1
        """
        # Convert to numpy for sklearn
        features_np = features.detach().cpu().numpy()
        labels_np = cluster_labels.detach().cpu().numpy()

        # Compute silhouette score if we have at least 2 clusters with data
        unique_labels = np.unique(labels_np)
        if len(unique_labels) > 1:
            score = silhouette_score(features_np, labels_np, metric='cosine')
            return score
        return 0.0

    def find_optimal_subcentroids(self, features, class_idx):
        """Find optimal number of subcentroids based on silhouette score.

        Args:
            features (torch.Tensor): Feature embeddings for one class
            class_idx (int): Class index
        Returns:
            int: Optimal number of subcentroids
            float: Best silhouette score
        """
        best_score = -1
        best_k = 4  # Default value

        # Only proceed if we have enough samples
        if features.shape[0] > max(self.candidate_subcentroids):
            for k in self.candidate_subcentroids:
                # Try clustering with k subcentroids
                init_q = torch.mm(features, self.prototypes[class_idx, :k, :].t())
                q, indexs = agd_torch_no_grad_gpu(init_q)

                # Compute silhouette score
                score = self.compute_silhouette(features, indexs)

                if score > best_score:
                    best_score = score
                    best_k = k
        return best_k, best_score

    def subcentroids_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        # compute cosine similarity using all subcentroids
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        # compute logits and apply temperature
        centroid_logits = cosine_similarity / self.temperature
        centroid_target = gt_seg.clone().float()

        # clustering for each class
        centroids = self.prototypes.data.clone()

        for k in range(self.num_classes):
            # Get features for this class
            c_k = _c[gt_seg == k, ...]
            if c_k.shape[0] == 0:
                continue

            optimal_k, score = self.find_optimal_subcentroids(c_k, k)
            if score > self.best_silhouette[k]:
                self.optimal_subcentroids[k] = torch.tensor(optimal_k)
                self.best_silhouette[k] = torch.tensor(score)

            # Use optimal number of subcentroids for this class
            num_k = self.optimal_subcentroids[k]
            num_k = int(num_k.item())

            # get initial assignments for the k-th class
            init_q = masks[gt_seg == k, :num_k, k]

            # clustering
            q, indexes = agd_torch_no_grad_gpu(init_q)

            m_k = mask[gt_seg == k]
            m_k_tile = repeat(m_k, 'n -> n tile', tile=num_k)
            m_q = q * m_k_tile

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile

            f = m_q.transpose(0, 1) @ c_q
            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_subcentroids:
                f = F.normalize(f, p=2, dim=-1)
                new_value = self.momentum_update(
                    old_value=centroids[k, :num_k][n != 0],
                    new_value=f[n != 0],
                    momentum=self.gamma
                )
                centroids[k, :num_k][n != 0] = new_value

            # Update target indices
            centroid_target[gt_seg == k] = indexes.float() + (num_k * k)

        # Update prototypes
        self.prototypes = nn.Parameter(F.normalize(centroids, p=2, dim=-1),
                                       requires_grad=self.pretrain_subcentroids)

        # Sync across GPUs
        if self.use_subcentroids and dist.is_available() and dist.is_initialized():
            centroids = self.prototypes.data.clone()
            dist.all_reduce(centroids.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(centroids, requires_grad=self.pretrain_subcentroids)

        return centroid_logits, centroid_target

    def forward_train(self, x, gt_label, **kwargs):
        inputs = self.pre_logits(x)  # (batch_size x 512)

        batch_size = inputs.shape[0]

        if self.is_only_cross_entropy:
            # Save to memory bank
            batch_size_num = self.dequeue_and_enqueue(inputs, gt_label, batch_size)

            if batch_size_num >= self.batch_size_num_limit:
                self.is_only_cross_entropy = False
            else:
                seg_logits = self.forward(inputs, gt_label)
                losses = self.loss(seg_logits, gt_label)

        if not self.pretrain_subcentroids and self.use_subcentroids and not self.is_only_cross_entropy:
            # concat data from memory_bank
            inputs = torch.cat(self.memory_bank, 0)
            gt_label = torch.cat(self.memory_bank_label, 0)

            seg_logits, contrast_logits, contrast_target = self.forward(inputs, gt_label=gt_label)
            losses = self.loss(seg_logits, gt_label, **kwargs)

            # release memory bank space
            self.memory_bank = []
            self.memory_bank_label = []

            if self.centroid_contrast_loss is True and self.is_only_cross_entropy is False:  # changes here: and self.isOnlyCE is False: # Only happens apply once.
                loss_centroid_contrast = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=255)
                losses['loss_centroid_contrast'] = loss_centroid_contrast * self.centroid_contrast_loss_weights

            # initialize isOnlyCE to True
            self.is_only_cross_entropy = True
        return losses

    def forward(self, x, img_metas=None, gt_label=None):
        x = self.feat_norm(x)
        x = F.normalize(x, p=2, dim=-1)

        self.prototypes.data.copy_(F.normalize(self.prototypes, p=2, dim=-1))

        # Create mask for optimal subcentroids
        optimal_mask = torch.zeros_like(self.prototypes)
        for k in range(self.num_classes):
            num_k = int(self.optimal_subcentroids[k].item())
            optimal_mask[k, :num_k, :] = 1.0

        # Apply mask to prototypes
        masked_prototypes = self.prototypes * optimal_mask

        # Compute masks using masked prototypes
        masks = torch.einsum('nd,kmd->nmk', x, masked_prototypes)  # originally nmk

        out_cls = torch.amax(masks, dim=1)

        out_cls = self.mask_norm(out_cls)

        if not self.pretrain_subcentroids and self.use_subcentroids and gt_label is not None and not self.is_only_cross_entropy:
            contrast_logits, contrast_target = self.subcentroids_learning(x, out_cls, gt_label, masks)
            return out_cls, contrast_logits, contrast_target
        else:
            return out_cls

    def dequeue_and_enqueue(self, inputs, gt_label, batch_size):
        inputs_all = concat_all_gather(inputs)
        label_all = concat_all_gather(gt_label)

        self.memory_bank.append(inputs_all)
        self.memory_bank_label.append(label_all)

        return len(self.memory_bank)


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
