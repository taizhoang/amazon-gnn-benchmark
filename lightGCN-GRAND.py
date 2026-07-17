import argparse
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

p = argparse.ArgumentParser()
p.add_argument('--K', type=int, default=4)            # so hop propagation
p.add_argument('--hidden', type=int, default=64)
p.add_argument('--dropnode', type=float, default=0.5) # ti le DropNode
p.add_argument('--dropout', type=float, default=0.5)  # dropout trong MLP
p.add_argument('--S', type=int, default=2)            # so augmentation/step
p.add_argument('--lam', type=float, default=1.0)      # trong so consistency
p.add_argument('--temp', type=float, default=0.5)     # sharpening temperature
p.add_argument('--lr', type=float, default=0.01)
p.add_argument('--l2', type=float, default=5e-4)
p.add_argument('--epochs', type=int, default=250)
p.add_argument('--patience', type=int, default=50)
p.add_argument('--learn_gamma', type=int, default=1)
p.add_argument('--seeds', type=int, default=10)
p.add_argument('--prop_space', default='logits', choices=['logits','hidden'])
p.add_argument('--use_H', type=int, default=1)     # kenh compatibility (CoP)
p.add_argument('--Kc', type=int, default=2)        # so hop kenh H
p.add_argument('--conf', type=float, default=0.7)     # nguong confidence mask
p.add_argument('--warmup', type=float, default=60.0)  # epochs warmup lambda
p.add_argument('--min_epochs', type=int, default=120) # khong early-stop truoc moc nay
args = p.parse_args()

# ----------------------------- data ---------------------------------
d = np.load('data/Computers/raw/amazon_electronics_computers.npz', allow_pickle=True)
A = sp.csr_matrix((d['adj_data'], d['adj_indices'], d['adj_indptr']),
                  shape=d['adj_shape'])
A = A.maximum(A.T); A.data[:] = 1
X_np = np.asarray(sp.csr_matrix(
    (d['attr_data'], d['attr_indices'], d['attr_indptr']),
    shape=d['attr_shape']).todense(), dtype=np.float32)
y_np = d['labels'].astype(np.int64)
N, IN_DIM = X_np.shape
C = int(y_np.max()) + 1

M = A + sp.eye(N)                       # self-loop (bat buoc, da ablation)
deg = np.asarray(M.sum(1)).ravel()
Dm = sp.diags(deg ** -0.5)
Mn = (Dm @ M @ Dm).tocoo()
A_hat = tf.sparse.reorder(tf.sparse.SparseTensor(
    np.vstack([Mn.row, Mn.col]).T, Mn.data.astype(np.float32), Mn.shape))
Mn_sp = (Dm @ M @ Dm).tocsr()              # ban scipy cho teacher
_E = sp.triu(A, k=1).tocoo()
E_row, E_col = _E.row, _E.col              # edge list cho uoc luong H
X = tf.constant(X_np)
y = tf.constant(y_np)


# ----------------------------- model --------------------------------
class LightGCN_GRAND(tf.Module):
    def __init__(self):
        super().__init__()
        def glorot(shape, seed):
            # moi bien 1 initializer co seed rieng — tranh loi Keras tra
            # gia tri trung nhau khi tai su dung 1 instance unseeded
            return tf.keras.initializers.GlorotUniform(seed=seed)(shape)
        self.W1 = tf.Variable(glorot([IN_DIM, args.hidden], args.var_seed))
        self.b1 = tf.Variable(tf.zeros([args.hidden]))
        self.W2 = tf.Variable(glorot([args.hidden, C], args.var_seed + 1))
        self.b2 = tf.Variable(tf.zeros([C]))
        # gamma: layer-combination weights (LightGCN: co dinh 1/(K+1))
        self.gamma = tf.Variable(tf.zeros([args.K + 1]),
                                 trainable=bool(args.learn_gamma))
        if args.use_H:
            # Kenh compatibility: H init tu teacher, hoc tiep; gate mix
            self.H = tf.Variable(args.H_init)          # 10x10
            self.delta = tf.Variable(tf.zeros([args.Kc]))
            self.mix = tf.Variable(-2.0)               # sigmoid(-2)~0.12

    def propagate(self, Xin):
        """Core LightGCN: khong transform, khong nonlinearity giua layer."""
        g = tf.nn.softmax(self.gamma)
        ego = Xin
        out = g[0] * ego
        for k in range(1, args.K + 1):
            ego = tf.sparse.sparse_dense_matmul(A_hat, ego)
            out += g[k] * ego
        return out

    def __call__(self, Xin, training):
        if training and args.dropnode > 0:
            # DropNode (GRAND): tat ngau nhien toan bo feature cua 1 node,
            # scale 1/(1-p) de giu ky vong
            mask = tf.cast(tf.random.uniform([N, 1]) > args.dropnode,
                           tf.float32) / (1.0 - args.dropnode)
            Xin = Xin * mask
        # MLP truoc, roi propagate LOGITS (10 chieu) — thu tu APPNP/GPR-GNN,
        # re hon 70x so voi propagate 767 chieu ma cung ho decoupled;
        # propagation van la core LightGCN: khong transform, weighted sum hop
        h = tf.nn.relu(tf.matmul(Xin, self.W1) + self.b1)
        if training:
            h = tf.nn.dropout(h, args.dropout)
        if args.prop_space == 'hidden':
            # Propagate HIDDEN (64d): augmentation da dang hon logits 10d,
            # gan GRAND goc hon — nguon diem con lai cua kien truc
            h = self.propagate(h)
            return tf.matmul(h, self.W2) + self.b2
        logits0 = tf.matmul(h, self.W2) + self.b2
        zA = self.propagate(logits0)
        if not args.use_H:
            return zA
        # Kenh CoP: truyen phan phoi lop qua ma tran tuong thich H moi hop
        Hn = tf.nn.softmax(self.H, axis=1)
        P = tf.nn.softmax(logits0)
        dl = tf.nn.softmax(self.delta)
        ego = P; zB = tf.zeros_like(P)
        for k in range(args.Kc):
            ego = tf.sparse.sparse_dense_matmul(A_hat, tf.matmul(ego, Hn))
            zB += dl[k] * ego
        m = tf.nn.sigmoid(self.mix)
        return zA + m * tf.math.log(zB + 1e-8)


def make_split(seed, n_tr=20, n_va=30):
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in range(C):
        idx = rng.permutation(np.where(y_np == c)[0])
        tr += list(idx[:n_tr]); va += list(idx[n_tr:n_tr + n_va])
        te += list(idx[n_tr + n_va:])
    return map(np.array, (tr, va, te))


# Precompute mean-propagated features cho teacher (dung chung moi seed)
_F_teach = None
def teacher_H_init(tr):
    """Uoc luong ma tran chuyen lop H tu pseudo-label cua teacher SGC:
    logreg tren mean_k A^k X -> gan nhan toan bo -> dem cap lop tren edges."""
    global _F_teach
    from sklearn.linear_model import LogisticRegression
    if _F_teach is None:
        Hk = X_np.copy(); acc = X_np.copy()
        import scipy.sparse as _sp
        for _ in range(2):
            Hk = Mn_sp @ Hk; acc = acc + Hk
        _F_teach = acc / 3.0
    clf = LogisticRegression(max_iter=1500, C=10.).fit(_F_teach[tr], y_np[tr])
    pl = clf.predict(_F_teach)
    pairs = np.zeros((C, C), dtype=np.float64)
    np.add.at(pairs, (pl[E_row], pl[E_col]), 1.0)
    pairs = pairs + pairs.T
    pairs = pairs / pairs.sum(1, keepdims=True)
    return pairs.astype(np.float32)


def run_seed(seed):
    tf.random.set_seed(seed)
    args.var_seed = 1000 + seed * 7
    tr, va, te = make_split(seed)
    tr_t = tf.constant(tr); y_tr = tf.constant(y_np[tr])
    if args.use_H:
        args.H_init = teacher_H_init(tr)
    model = LightGCN_GRAND()
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function
    def train_step(lam):
        with tf.GradientTape() as tape:
            # S lan forward voi DropNode khac nhau
            probs = [tf.nn.softmax(model(X, training=True))
                     for _ in range(args.S)]
            # Supervised CE (trung binh tren S augmentation)
            sup = tf.add_n([
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_tr,
                    logits=tf.math.log(tf.gather(pr, tr_t) + 1e-8)))
                for pr in probs]) / args.S
            # Consistency (GRAND): sharpen trung binh du doan roi keo cac
            # augmentation ve target — tinh tren TOAN BO node (ke ca unlabeled)
            avg = tf.add_n(probs) / args.S
            sharp = tf.pow(avg, 1.0 / args.temp)
            sharp = tf.stop_gradient(
                sharp / tf.reduce_sum(sharp, 1, keepdims=True))
            # Confidence mask: chi ap consistency len node co du doan tu tin
            conf_mask = tf.stop_gradient(tf.cast(
                tf.reduce_max(avg, 1) > args.conf, tf.float32))
            denom = tf.reduce_sum(conf_mask) + 1e-8
            con = tf.add_n([tf.reduce_sum(
                conf_mask * tf.reduce_sum(tf.square(pr - sharp), 1)) / denom
                for pr in probs]) / args.S
            l2 = args.l2 * (tf.nn.l2_loss(model.W1) + tf.nn.l2_loss(model.W2))
            loss = sup + lam * con + l2
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    @tf.function
    def predict():
        return tf.argmax(model(X, training=False), 1)

    best_va, best_te, wait = 0., 0., 0
    for ep in range(args.epochs):
        lam = args.lam * min(1.0, ep / args.warmup)  # warmup consistency
        train_step(tf.constant(lam, tf.float32))
        pred = predict().numpy()
        acc_va = (pred[va] == y_np[va]).mean()
        if acc_va > best_va:
            best_va, best_te, wait = acc_va, (pred[te] == y_np[te]).mean(), 0
        else:
            wait += 1
            if ep >= args.min_epochs and wait >= args.patience:
                break
    return best_te


if __name__ == '__main__':
    from sklearn.metrics import f1_score as _f1  # noqa
    accs = [run_seed(s) for s in range(args.seeds)]
    for s, a in enumerate(accs):
        print(f'seed {s}: test acc {a*100:.2f}')
    print(f'\nLightGCN-GRAND K={args.K} S={args.S} dropnode={args.dropnode} '
          f'lam={args.lam} learn_gamma={args.learn_gamma}')
    print(f'Test acc ({args.seeds} seeds): '
          f'{np.mean(accs)*100:.2f} +- {np.std(accs)*100:.2f}')
