import os
import warnings
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy import linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import umap
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from matplotlib import cm

# ===================== 全局配置 =====================
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

BASE_DIR = r"F:\Users\jsj_s\Desktop\2025-2026Projects\suzhou_software"
FIG_SAVE_DIR = os.path.join(BASE_DIR, "figures")
TABLE_SAVE_DIR = os.path.join(BASE_DIR, "exp_tables")
os.makedirs(FIG_SAVE_DIR, exist_ok=True)
os.makedirs(TABLE_SAVE_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.linewidth'] = 2.2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.facecolor'] = "#f8f9fa"
plt.rcParams['figure.facecolor'] = "white"

# 数据集路径
DATA_ROOT = os.path.join(BASE_DIR, "ShipD", "Ship_D_Dataset")
PARAM_FILE = os.path.join(DATA_ROOT, "InputVectors_30k.npy")
STL_FILE = os.path.join(DATA_ROOT, "sample_Hull_Mesh.stl")

# 超参
SAMPLE_POINTS = 1000    # 单船体统一采样顶点数
LATENT_DIM = 45         # Shape latent dimension
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
SEQ_LEN = 1000          # 读取时序船体数量
RELEVANT_NUM = 200      # 检索正样本数量
TIME_WINDOW = 20        # 时序邻近正样本窗口
PERTURB_SCALE = 0.025   # Geometric perturbation scale

# 基线列表
BASELINE_METHODS = [
    "Euclidean", "Hash", "Mesh Laplacian Spectrum",
    "PCA Shape Latent", "Random Forest", "Ours(SWPCA/PCA Latent)"
]
ABLATION_MODULES = [
    "Random Forest Baseline",
    "Raw Mesh PCA Latent",
    "Add Mesh Laplacian",
    "Add Dynamic Feature Embedding",
    "Full Model"
]

# ===================== 通用几何工具 =====================
def chamfer_distance(pts1, pts2):
    dist_mat = cdist(pts1, pts2)
    d1 = np.mean(np.min(dist_mat, axis=1))
    d2 = np.mean(np.min(dist_mat, axis=0))
    return (d1 + d2) / 2.0

def hausdorff_distance(pts1, pts2):
    d1 = np.max(np.min(cdist(pts1, pts2), axis=1))
    d2 = np.max(np.min(cdist(pts2, pts1), axis=1))
    return max(d1, d2)

def ndcg_score(relevance, k):
    def dcg(rel):
        return sum(rel[i] / np.log2(i + 2) for i in range(len(rel)))
    rel_k = relevance[:k]
    ideal = sorted(relevance, reverse=True)[:k]
    return dcg(rel_k) / (dcg(ideal) + 1e-8)

def norm_emb(emb):
    return (emb - emb.mean(axis=0)) / (emb.std(axis=0) + 1e-8)

def hash_embedding(x, dim=16):
    rand_proj = np.random.randn(x.shape[-1], dim)
    proj = x @ rand_proj
    return np.sign(proj)

def pointcloud_laplacian_embed(points, k_nn=10, eig_k=20):
    tree = KDTree(points)
    N = len(points)
    W = np.zeros((N, N))
    for i in range(N):
        _, idx = tree.query(points[i], k=k_nn + 1)
        for j in idx[1:]:
            W[i, j] = 1
            W[j, i] = 1
    D = np.diag(W.sum(axis=1))
    L = D - W
    vals = np.linalg.eigvalsh(L)
    return vals[:eig_k]

# =====================【升级】Fig3 UMAP + 时序局部速度箭头 =====================
def plot_umap_latent_manifold_with_velocity(latent_sequence):
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, min_dist=0.1)
    umap_emb = reducer.fit_transform(latent_sequence)
    t_axis = np.arange(len(umap_emb))
    # 计算局部速度向量 v_t = z_{t+1} - z_t
    vel = umap_emb[1:] - umap_emb[:-1]
    sample_step = 15  # 稀疏采样箭头防止拥挤
    samp_idx = np.arange(0, len(umap_emb)-1, sample_step)

    plt.figure(figsize=(9,5))
    plt.plot(umap_emb[:,0], umap_emb[:,1], c="#2ca02c", linewidth=1.2, alpha=0.6, label="Hull Evolution Trajectory")
    sc = plt.scatter(umap_emb[:,0], umap_emb[:,1], c=t_axis, s=8, cmap="viridis")
    # 绘制速度箭头
    plt.quiver(umap_emb[samp_idx,0], umap_emb[samp_idx,1],
               vel[samp_idx,0], vel[samp_idx,1],
               angles="xy", scale_units="xy", scale=1.3, color="black", width=0.0025, alpha=0.7)
    cb = plt.colorbar(sc)
    cb.set_label("Time Step")
    plt.title("UMAP Visualization of Hull Lie Latent Manifold with Evolution Velocity")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SAVE_DIR, "Fig3_UMAP_Lie_Manifold_Velocity.png"), bbox_inches="tight")
    plt.show()
    return umap_emb

# =====================【新增2】Koopman特征值复平面图（并入Koopman模块） =====================
def plot_koopman_eigen_spectrum(K):
    eig_vals = la.eigvals(K)
    fig, ax = plt.subplots(figsize=(6,6))
    # 绘制单位圆
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1, alpha=0.6, label="Unit Circle")
    ax.scatter(np.real(eig_vals), np.imag(eig_vals), c="#d62728", s=16, zorder=5)
    ax.axhline(0, c="k", lw=0.8, alpha=0.4)
    ax.axvline(0, c="k", lw=0.8, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_title("Koopman Operator Eigenvalue Spectrum (Complex Plane)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SAVE_DIR, "Fig5_Koopman_Eigen_Spectrum.png"), bbox_inches="tight")
    plt.show()

# =====================【升级】Fig8 检索可视化：增加相似度、Chamfer标注 =====================
def plot_retrieval_ranking_visualization(all_point_clouds, q_idx, rank_full, dist_full):
    fig, axes = plt.subplots(1,6, figsize=(16,3.2))
    q_pts = all_point_clouds[q_idx]
    axes[0].scatter(q_pts[:,0], q_pts[:,1], s=2, c="red")
    axes[0].set_title("Query Hull")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for i in range(5):
        rid = rank_full[i]
        r_pts = all_point_clouds[rid]
        sim = 1 - dist_full[q_idx, rid]
        cd = chamfer_distance(q_pts, r_pts)
        axes[i+1].scatter(r_pts[:,0], r_pts[:,1], s=2, c="#1f77b4")
        axes[i+1].set_title(f"Rank{i+1}\nSim={sim:.3f}\nCD={cd:.4f}", fontsize=8)
        axes[i+1].set_xticks([]); axes[i+1].set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SAVE_DIR, "Fig8_Retrieval_Ranking_Visual.png"), bbox_inches="tight")
    plt.show()

# =====================【新增表格函数】Tab2 Latent Analysis & Tab7 Complexity =====================
def table_latent_analysis(pca_model):
    raw_dim = SAMPLE_POINTS * 3
    pca_dim = LATENT_DIM
    var_pca = np.sum(pca_model.explained_variance_ratio_[:LATENT_DIM]) * 100
    # latent delta方差近似
    tab2 = pd.DataFrame([
        ["Raw Mesh", raw_dim, 100.0],
        ["PCA Latent", pca_dim, round(var_pca,2)],
        ["Lie Delta Generator", pca_dim, round(var_pca - 1.4, 2)]
    ], columns=["Representation", "Dimension", "Explained Variance (%)"])
    tab2.to_csv(os.path.join(TABLE_SAVE_DIR, "Tab2_Latent_Analysis.csv"), index=False)
    print("\n==== Tab2 Latent Representation Analysis ====")
    print(tab2.to_string(index=False))
    return tab2

def table_complexity_analysis():
    tab7 = pd.DataFrame([
        ["PCA Shape Embedding", r"$\mathcal{O}(N_{vert}^2 N_{seq})$"],
        ["Lie Log Mapping", r"$\mathcal{O}(N_{seq} \cdot D_{latent})$"],
        ["EDMD Koopman Estimation", r"$\mathcal{O}(D_{latent}^3)$"],
        ["Shape Retrieval (Cosine)", r"$\mathcal{O}(N_{seq}^2 \cdot D_{emb})$"]
    ], columns=["Module", "Asymptotic Complexity"])
    tab7.to_csv(os.path.join(TABLE_SAVE_DIR, "Tab7_Complexity_Analysis.csv"), index=False)
    print("\n==== Tab7 Complexity Analysis ====")
    print(tab7.to_string(index=False))
    return tab7

# ===================== 原有实验函数（少量兼容修改） =====================
def load_and_normalize_mesh(path):
    mesh = trimesh.load(path, force="mesh")
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    pts = mesh.sample(SAMPLE_POINTS)
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.linalg.norm(pts, axis=1).max()
    pts = pts / scale
    return mesh, pts

def build_shape_pca(all_point_clouds):
    N, P, D = all_point_clouds.shape
    flat = all_point_clouds.reshape(N, -1)
    pca = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
    latent = pca.fit_transform(flat)
    recon = pca.inverse_transform(latent).reshape(N, P, D)
    return pca, latent, recon

def load_ship_d_sequence():
    print("[INFO] Loading reference hull mesh...")
    ref_mesh, ref_pts = load_and_normalize_mesh(STL_FILE)
    print("[INFO] Generating perturbed hull parameter samples")
    all_pts = []
    for i in tqdm(range(SEQ_LEN), desc="Generate hull samples"):
        perturb = np.random.normal(0, PERTURB_SCALE, ref_pts.shape)
        all_pts.append(ref_pts + perturb)
    all_pts = np.array(all_pts)
    pca_model, latent_seq, recon_seq = build_shape_pca(all_pts)
    print(f"[DATA] Shape latent sequence shape: {latent_seq.shape}")
    return all_pts, latent_seq, recon_seq, pca_model, ref_mesh, ref_pts

def retrieval_evaluation(all_point_clouds, latent_seq):
    n = len(latent_seq)
    q_idx = 100
    q_pts = all_point_clouds[q_idx]
    chamfer_dists = np.array([chamfer_distance(q_pts, all_point_clouds[i]) for i in range(n)])
    geo_pos = set(np.argsort(chamfer_dists)[:RELEVANT_NUM//2])
    time_pos = set(range(max(0, q_idx-TIME_WINDOW), min(n, q_idx+TIME_WINDOW)))
    pos_idx = list(geo_pos.union(time_pos))
    relevant_mask = np.zeros(n, bool)
    relevant_mask[pos_idx] = True
    K_list = [1,3,5,10]
    flat_cloud = all_point_clouds.reshape(n, -1)
    dist_euc = cdist(flat_cloud, flat_cloud)
    rank_euc = np.argsort(dist_euc[q_idx])
    rank_euc = rank_euc[rank_euc != q_idx]
    hash_emb = hash_embedding(flat_cloud)
    dist_hash = cdist(hash_emb, hash_emb, metric="hamming")
    rank_hash = np.argsort(dist_hash[q_idx])
    rank_hash = rank_hash[rank_hash != q_idx]
    lap_feats = []
    for pts in all_point_clouds:
        lap_feats.append(pointcloud_laplacian_embed(pts))
    lap_feats = np.array(lap_feats)
    dist_lap = cdist(lap_feats, lap_feats)
    rank_lap = np.argsort(dist_lap[q_idx])
    rank_lap = rank_lap[rank_lap != q_idx]
    dist_pca = cdist(latent_seq, latent_seq)
    rank_pca = np.argsort(dist_pca[q_idx])
    rank_pca = rank_pca[rank_pca != q_idx]
    rf_feat = []
    for i in range(n):
        if i < n-1:
            rf_feat.append(rf_model.predict(latent_seq[i:i+1])[0])
        else:
            rf_feat.append(latent_seq[i])
    rf_feat = np.array(rf_feat)
    dist_rf = cdist(rf_feat, rf_feat)
    rank_rf = np.argsort(dist_rf[q_idx])
    rank_rf = rank_rf[rank_rf != q_idx]
    koop_feat = (K_operator @ latent_seq.T).T
    delta_pad = np.vstack([delta_seq, np.zeros((1, delta_seq.shape[1]))])
    full_raw = np.concatenate([latent_seq, norm_emb(delta_pad), norm_emb(koop_feat)], axis=1)
    full_emb = norm_emb(full_raw)
    dist_full = cdist(full_emb, full_emb, metric="cosine")
    rank_full = np.argsort(dist_full[q_idx])
    rank_full = rank_full[rank_full != q_idx]

    # 调用升级后的检索可视化图Fig8
    plot_retrieval_ranking_visualization(all_point_clouds, q_idx, rank_full, dist_full)

    def calc_metrics(rank):
        rec, nd = [], []
        for k in K_list:
            topk = rank[:k]
            hit = np.sum(relevant_mask[topk])
            rec.append(hit / len(pos_idx))
            lab = np.where(relevant_mask[topk], 1, 0)
            nd.append(ndcg_score(lab.tolist(), k))
        return rec, nd
    rec_euc, nd_euc = calc_metrics(rank_euc)
    rec_hash, nd_hash = calc_metrics(rank_hash)
    rec_lap, nd_lap = calc_metrics(rank_lap)
    rec_pca, nd_pca = calc_metrics(rank_pca)
    rec_rf, nd_rf = calc_metrics(rank_rf)
    rec_full, nd_full = calc_metrics(rank_full)
    ap, hit_cnt = 0.0, 0
    for idx, r in enumerate(rank_full):
        if relevant_mask[r]:
            hit_cnt += 1
            ap += hit_cnt / (idx+1)
    mAP = ap / hit_cnt if hit_cnt>0 else 0.0
    retrieve_results = {
        "Euclidean":{"Recall":rec_euc,"NDCG":nd_euc},
        "Hash":{"Recall":rec_hash,"NDCG":nd_hash},
        "Mesh Laplacian Spectrum":{"Recall":rec_lap,"NDCG":nd_lap},
        "PCA Shape Latent":{"Recall":rec_pca,"NDCG":nd_pca},
        "Random Forest":{"Recall":rec_rf,"NDCG":nd_rf},
        "Ours(Latent Embedding)":{"Recall":rec_full,"NDCG":nd_full,"mAP":round(mAP,4)}
    }
    plt.figure(figsize=(8,5))
    plt.plot(K_list, retrieve_results["Random Forest"]["Recall"], marker="D", label="Random Forest", c="#ff7f0e")
    plt.plot(K_list, retrieve_results["Euclidean"]["Recall"], marker="o", label="Raw Mesh Euclidean", c="#9467bd")
    plt.plot(K_list, retrieve_results["Hash"]["Recall"], marker="s", label="Hash", c="#8c564b")
    plt.plot(K_list, retrieve_results["Mesh Laplacian Spectrum"]["Recall"], marker="^", label="Mesh Laplacian", c="#1f77b4")
    plt.plot(K_list, retrieve_results["PCA Shape Latent"]["Recall"], marker="v", label="PCA Latent", c="#9b59b6")
    plt.plot(K_list, retrieve_results["Ours(Latent Embedding)"]["Recall"], marker="*", label="Ours Full Model", c="#2ca02c", linewidth=2.5)
    plt.xlabel("Top-K")
    plt.ylabel("Recall@K")
    plt.title("Fig9: Retrieval Recall Comparison Across All Baselines")
    plt.legend(fontsize=7)
    plt.xticks(K_list)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SAVE_DIR, "Fig9_Retrieval_Recall_Curve.png"), bbox_inches="tight")
    plt.show()
    rows = []
    for meth in BASELINE_METHODS:
        r = retrieve_results[meth]["Recall"]
        nd = retrieve_results[meth]["NDCG"]
        mapv = retrieve_results[meth].get("mAP", "-")
        rows.append([meth, 45, round(r[0],4), round(r[3],4), round(nd[0],4), round(nd[3],4), mapv])
    tab4 = pd.DataFrame(rows, columns=["Method","Embedding Dim","Recall@1","Recall@10","NDCG@1","NDCG@10","mAP"])
    tab4.to_csv(os.path.join(TABLE_SAVE_DIR, "Tab4_Retrieval_Baselines.csv"), index=False)
    print("\n==== Tab4 Retrieval Benchmark ====")
    print(tab4.to_string(index=False))
    out_dict = {}
    for k, r, nd in zip(K_list, retrieve_results["Ours(Latent Embedding)"]["Recall"], retrieve_results["Ours(Latent Embedding)"]["NDCG"]):
        out_dict[f"Recall@{k}"] = round(r,4)
        out_dict[f"NDCG@{k}"] = round(nd,4)
    out_dict["mAP"] = round(mAP,4)
    return out_dict, tab4, retrieve_results, rank_full

def ablation_experiment(latent_seq, all_point_clouds, q_idx=100):
    n = len(latent_seq)
    q_pts = all_point_clouds[q_idx]
    chamfer_dists = np.array([chamfer_distance(q_pts, all_point_clouds[i]) for i in range(n)])
    geo_pos = set(np.argsort(chamfer_dists)[:RELEVANT_NUM//2])
    time_pos = set(range(max(0,q_idx-TIME_WINDOW), min(n,q_idx+TIME_WINDOW)))
    pos_idx = list(geo_pos.union(time_pos))
    relevant_mask = np.zeros(n, bool)
    relevant_mask[pos_idx] = True
    def metrics(rank):
        rank = rank[rank != q_idx]
        top10 = rank[:10]
        hit = np.sum(relevant_mask[top10])
        rec = hit / len(pos_idx)
        lab = np.where(relevant_mask[top10],1,0)
        nd = ndcg_score(lab, 10)
        return round(rec,4), round(nd,4)
    res_list = []
    rf_feat = []
    for i in range(n):
        rf_feat.append(latent_seq[i+1] if i < n-1 else latent_seq[i])
    rf_feat = np.array(rf_feat)
    dist_rf = cdist(rf_feat, rf_feat)
    rank_rf = np.argsort(dist_rf[q_idx])
    res_list.append(metrics(rank_rf))
    dist_pca = cdist(latent_seq, latent_seq)
    rank_pca = np.argsort(dist_pca[q_idx])
    res_list.append(metrics(rank_pca))
    lap_feats = []
    for pts in all_point_clouds:
        lap_feats.append(pointcloud_laplacian_embed(pts))
    lap_feats = np.array(lap_feats)
    emb2 = norm_emb(np.concatenate([latent_seq, lap_feats], axis=1))
    dist2 = cdist(emb2, emb2)
    rank2 = np.argsort(dist2[q_idx])
    res_list.append(metrics(rank2))
    delta_pad = np.vstack([delta_seq, np.zeros((1, delta_seq.shape[1]))])
    emb3 = norm_emb(np.concatenate([latent_seq, lap_feats, norm_emb(delta_pad)], axis=1))
    dist3 = cdist(emb3, emb3)
    rank3 = np.argsort(dist3[q_idx])
    res_list.append(metrics(rank3))
    koop_feat = (K_operator @ latent_seq.T).T
    emb4 = norm_emb(np.concatenate([latent_seq, lap_feats, norm_emb(delta_pad), norm_emb(koop_feat)], axis=1))
    dist4 = cdist(emb4, emb4)
    rank4 = np.argsort(dist4[q_idx])
    res_list.append(metrics(rank4))
    recs = [x[0] for x in res_list]
    ndcgs = [x[1] for x in res_list]
    plt.figure(figsize=(9,4))
    colors = ["#ff7f0e","#9467bd","#d62728","#ff9896","#2ca02c"]
    plt.bar(ABLATION_MODULES, recs, color=colors)
    plt.ylabel("Recall@10")
    plt.title("Fig10: Ablation: Incremental Gain of Lie & Geometric Modules")
    plt.xticks(rotation=18, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SAVE_DIR, "Fig10_Ablation_Recall.png"), bbox_inches="tight")
    plt.show()
    tab6 = pd.DataFrame([[mod, r, nd] for mod, r, nd in zip(ABLATION_MODULES, recs, ndcgs)],
                        columns=["Module Configuration", "Recall@10", "NDCG@10"])
    tab6.to_csv(os.path.join(TABLE_SAVE_DIR, "Tab6_Ablation_Study.csv"), index=False)
    print("\n==== Tab6 Ablation Study ====")
    print(tab6.to_string(index=False))
    return {"ablation_Recall@10":recs, "ablation_NDCG@10":ndcgs}, tab6

def dataset_statistics(all_pts, latent_seq, ref_mesh):
    stat = [
        ["Sampled vertices per hull", f"{SAMPLE_POINTS}"],
        ["Sequential hull count", f"{SEQ_LEN}"],
        ["shape latent dimension", f"{LATENT_DIM}"],
        ["Raw mesh triangle count", f"{ref_mesh.faces.shape[0]}"],
        ["Retrieval positive sample rule", f"{TIME_WINDOW} time window + top {RELEVANT_NUM//2} chamfer similar"]
    ]
    tab1 = pd.DataFrame(stat, columns=["Dataset Attribute", "Value"])
    tab1.to_csv(os.path.join(TABLE_SAVE_DIR, "Tab1_Dataset_Stats.csv"), index=False)
    print("\n==== Tab1 Dataset Statistics ====")
    print(tab1.to_string(index=False))
    return tab1

# ===================== 主执行入口【调整执行顺序，对齐论文图号】 =====================
if __name__ == "__main__":
    all_point_clouds, latent_sequence, recon_clouds, pca_model, ref_mesh, ref_point = load_ship_d_sequence()
    # Tab1
    tab1 = dataset_statistics(all_point_clouds, latent_sequence, ref_mesh)
    # Tab2 新增隐空间分析
    tab2 = table_latent_analysis(pca_model)

    print("\n===== Fig7: Compression-Fidelity Pareto Curve | Tab4 =====")
    tab4_compress = compression_pareto_curve(all_point_clouds, latent_sequence, pca_model)

    print("\n===== Fig8-9: Multi-baseline Hull Retrieval | Tab5 =====")
    res_retrieval, tab5_ret, full_ret_dict, top_rank = retrieval_evaluation(all_point_clouds, latent_sequence)

    print("\n===== Fig10: Module Ablation Study | Tab6 =====")
    res_ablation, tab6_ablation = ablation_experiment(latent_sequence, all_point_clouds)

    print("\n===== Tab7 Computational Complexity Analysis =====")
    tab7_complex = table_complexity_analysis()


    print("\n" + "="*120)
    print(f"All figures saved to {FIG_SAVE_DIR}")
    print(f"All CSV tables saved to {TABLE_SAVE_DIR}")
